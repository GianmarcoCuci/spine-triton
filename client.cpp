#include "http_client.h"

#include <H5Cpp.h>
#include <curl/curl.h>
#include <zip.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;
namespace tc = triton::client;

namespace {

constexpr const char* kDefaultUrl = "localhost:8000";
constexpr const char* kDefaultModelName = "spine_icarus_full_chain";
constexpr const char* kDefaultModelVersion = "1";

enum class ScalarType { Int32, Int64, Float32 };

struct TensorSpec {
  std::string archive_name;
  std::string triton_name;
  ScalarType type;
  int rank;
  std::optional<int64_t> columns;
};

const std::vector<TensorSpec> kInputSpecs = {
    {"coordinates", "COORDINATES", ScalarType::Int32, 2, 4},
    {"features", "FEATURES", ScalarType::Float32, 2, 8},
    {"sources", "SOURCES", ScalarType::Int32, 2, 2},
    {"counts", "COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"meta", "META", ScalarType::Float32, 2, 12},
    {"run_info", "RUN_INFO", ScalarType::Int64, 2, 3},
};

const std::vector<TensorSpec> kFlashInputSpecs = {
    {"flash_data", "FLASH_DATA", ScalarType::Float32, 2, 16},
    {"flash_pe", "FLASH_PE", ScalarType::Float32, 2, 180},
    {"flash_counts", "FLASH_COUNTS", ScalarType::Int32, 1, std::nullopt},
};


const std::vector<TensorSpec> kOutputSpecs = {
    {"", "EVENT_ID", ScalarType::Int64, 2, 3},
    {"", "GHOST_PRED", ScalarType::Int64, 1, std::nullopt},
    {"", "GHOST_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "SEGMENTATION", ScalarType::Int64, 1, std::nullopt},
    {"", "SEGMENTATION_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "ORIG_INDEX", ScalarType::Int64, 1, std::nullopt},
    {"", "POINTS", ScalarType::Float32, 2, 3},
    {"", "DEPOSITIONS", ScalarType::Float32, 1, std::nullopt},
    {"", "DEGHOSTED_SOURCES", ScalarType::Int32, 2, 2},
    {"", "PARTICLES", ScalarType::Float32, 2, 41},
    {"", "PARTICLE_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "PARTICLE_VOXELS", ScalarType::Int64, 2, 3},
    {"", "PARTICLE_VOXEL_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "INTERACTIONS", ScalarType::Float32, 2, 29},
    {"", "INTERACTION_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "INTERACTION_FLASHES", ScalarType::Float32, 2, 6},
    {"", "INTERACTION_FLASH_COUNTS", ScalarType::Int32, 1, std::nullopt},
    {"", "FLASH_MATCH_RAN", ScalarType::Int32, 1, std::nullopt},
};

using TensorStorage = std::variant<std::vector<int32_t>, std::vector<int64_t>,
                                   std::vector<float>>;

struct Tensor {
  ScalarType type;
  std::vector<int64_t> shape;
  TensorStorage storage;

  size_t element_count() const {
    return std::visit([](const auto& values) { return values.size(); }, storage);
  }

  size_t byte_size() const {
    return std::visit(
        [](const auto& values) { return values.size() * sizeof(values[0]); },
        storage);
  }

  const uint8_t* raw_data() const {
    return std::visit(
        [](const auto& values) -> const uint8_t* {
          if (values.empty()) {
            return nullptr;
          }
          return reinterpret_cast<const uint8_t*>(values.data());
        },
        storage);
  }
};

using TensorMap = std::map<std::string, Tensor>;

struct Arguments {
  fs::path input;
  std::optional<fs::path> output;
  std::string url = kDefaultUrl;
  std::string model_name = kDefaultModelName;
  std::string model_version = kDefaultModelVersion;
  bool skip_flash_match = false;
  double connection_timeout = 10.0;
  double network_timeout = 3600.0;
  int preview = 2;
  bool overwrite = false;
  bool verbose = false;
  bool help = false;
};

std::string scalar_name(ScalarType type) {
  switch (type) {
    case ScalarType::Int32:
      return "INT32";
    case ScalarType::Int64:
      return "INT64";
    case ScalarType::Float32:
      return "FP32";
  }
  throw std::logic_error("Unknown scalar type.");
}

size_t scalar_size(ScalarType type) {
  switch (type) {
    case ScalarType::Int32:
      return sizeof(int32_t);
    case ScalarType::Int64:
      return sizeof(int64_t);
    case ScalarType::Float32:
      return sizeof(float);
  }
  throw std::logic_error("Unknown scalar type.");
}

std::string expected_numpy_descr(ScalarType type) {
  switch (type) {
    case ScalarType::Int32:
      return "<i4";
    case ScalarType::Int64:
      return "<i8";
    case ScalarType::Float32:
      return "<f4";
  }
  throw std::logic_error("Unknown scalar type.");
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string join(const std::vector<std::string>& values,
                 const std::string& separator) {
  std::ostringstream stream;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      stream << separator;
    }
    stream << values[index];
  }
  return stream.str();
}

size_t checked_element_count(const std::vector<int64_t>& shape) {
  size_t result = 1;
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("A tensor shape contains a negative dimension.");
    }
    const size_t value = static_cast<size_t>(dimension);
    if (value != 0 && result > std::numeric_limits<size_t>::max() / value) {
      throw std::runtime_error("A tensor shape is too large.");
    }
    result *= value;
  }
  return result;
}

Tensor tensor_from_bytes(ScalarType type, std::vector<int64_t> shape,
                         const uint8_t* bytes, size_t byte_count) {
  const size_t element_count = checked_element_count(shape);
  if (element_count > std::numeric_limits<size_t>::max() / scalar_size(type) ||
      element_count * scalar_size(type) != byte_count) {
    throw std::runtime_error("Tensor byte count does not match its shape.");
  }

  auto copy_values = [bytes, byte_count](auto tag) -> TensorStorage {
    using Value = decltype(tag);
    std::vector<Value> values(byte_count / sizeof(Value));
    if (byte_count != 0) {
      std::memcpy(values.data(), bytes, byte_count);
    }
    return values;
  };

  TensorStorage storage;
  switch (type) {
    case ScalarType::Int32:
      storage = copy_values(int32_t{});
      break;
    case ScalarType::Int64:
      storage = copy_values(int64_t{});
      break;
    case ScalarType::Float32:
      storage = copy_values(float{});
      break;
  }
  return Tensor{type, std::move(shape), std::move(storage)};
}

template <typename Value>
const std::vector<Value>& values(const Tensor& tensor) {
  return std::get<std::vector<Value>>(tensor.storage);
}

const Tensor& at(const TensorMap& tensors, std::string_view name) {
  const auto iterator = tensors.find(std::string(name));
  if (iterator == tensors.end()) {
    throw std::logic_error("Internal error: missing tensor '" +
                           std::string(name) + "'.");
  }
  return iterator->second;
}

size_t rows(const Tensor& tensor) {
  if (tensor.shape.empty()) {
    throw std::logic_error("Internal error: scalar tensor has no rows.");
  }
  return static_cast<size_t>(tensor.shape[0]);
}

void validate_fixed_contract(const std::string& name, const Tensor& tensor,
                             const TensorSpec& spec) {
  if (tensor.type != spec.type) {
    throw std::runtime_error("Tensor '" + name + "' has datatype " +
                             scalar_name(tensor.type) + "; expected " +
                             scalar_name(spec.type) + ".");
  }
  if (tensor.shape.size() != static_cast<size_t>(spec.rank)) {
    throw std::runtime_error("Tensor '" + name + "' has rank " +
                             std::to_string(tensor.shape.size()) +
                             "; expected rank " + std::to_string(spec.rank) +
                             ".");
  }
  if (spec.columns.has_value() && tensor.shape.at(1) != *spec.columns) {
    throw std::runtime_error("Tensor '" + name +
                             "' has an invalid number of columns.");
  }
}

class NpzArchive {
 public:
  explicit NpzArchive(const fs::path& path) {
    int error_code = 0;
    archive_ = zip_open(path.string().c_str(), ZIP_RDONLY, &error_code);
    if (archive_ == nullptr) {
      zip_error_t error;
      zip_error_init_with_code(&error, error_code);
      const std::string detail = zip_error_strerror(&error);
      zip_error_fini(&error);
      throw std::runtime_error("Cannot open NPZ archive '" + path.string() +
                               "': " + detail);
    }
  }

  NpzArchive(const NpzArchive&) = delete;
  NpzArchive& operator=(const NpzArchive&) = delete;

  ~NpzArchive() {
    if (archive_ != nullptr) {
      zip_discard(archive_);
    }
  }

  bool contains(const std::string& name) const {
    return zip_name_locate(archive_, name.c_str(), 0) >= 0;
  }

  std::vector<uint8_t> read(const std::string& name) const {
    const zip_int64_t index = zip_name_locate(archive_, name.c_str(), 0);
    if (index < 0) {
      throw std::runtime_error("NPZ archive is missing '" + name + "'.");
    }

    zip_stat_t status;
    zip_stat_init(&status);
    if (zip_stat_index(archive_, static_cast<zip_uint64_t>(index), 0,
                       &status) != 0) {
      throw std::runtime_error("Cannot inspect NPZ member '" + name + "': " +
                               zip_strerror(archive_));
    }
    if ((status.valid & ZIP_STAT_SIZE) == 0 ||
        status.size > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("NPZ member '" + name + "' is too large.");
    }

    zip_file_t* member =
        zip_fopen_index(archive_, static_cast<zip_uint64_t>(index), 0);
    if (member == nullptr) {
      throw std::runtime_error("Cannot open NPZ member '" + name + "': " +
                               zip_strerror(archive_));
    }

    std::vector<uint8_t> result(static_cast<size_t>(status.size));
    size_t offset = 0;
    while (offset < result.size()) {
      const zip_int64_t count =
          zip_fread(member, result.data() + offset, result.size() - offset);
      if (count < 0) {
        const std::string detail = zip_file_strerror(member);
        zip_fclose(member);
        throw std::runtime_error("Cannot read NPZ member '" + name + "': " +
                                 detail);
      }
      if (count == 0) {
        zip_fclose(member);
        throw std::runtime_error("NPZ member '" + name +
                                 "' ended before its declared size.");
      }
      offset += static_cast<size_t>(count);
    }
    if (zip_fclose(member) != 0) {
      throw std::runtime_error("Cannot close NPZ member '" + name + "'.");
    }
    return result;
  }

 private:
  zip_t* archive_ = nullptr;
};

uint16_t read_little_u16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         (static_cast<uint16_t>(data[1]) << 8U);
}

uint32_t read_little_u32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8U) |
         (static_cast<uint32_t>(data[2]) << 16U) |
         (static_cast<uint32_t>(data[3]) << 24U);
}

std::vector<int64_t> parse_numpy_shape(const std::string& body,
                                       const std::string& member_name) {
  std::vector<int64_t> shape;
  std::stringstream stream(body);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = trim(token);
    if (token.empty()) {
      continue;
    }
    if (token.front() == '-') {
      throw std::runtime_error("NPY member '" + member_name +
                               "' has a negative dimension.");
    }
    size_t parsed = 0;
    unsigned long long dimension = 0;
    try {
      dimension = std::stoull(token, &parsed);
    } catch (const std::exception&) {
      throw std::runtime_error("Cannot parse the shape of NPY member '" +
                               member_name + "'.");
    }
    if (parsed != token.size() ||
        dimension > static_cast<unsigned long long>(
                        std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error("Invalid dimension in NPY member '" +
                               member_name + "'.");
    }
    shape.push_back(static_cast<int64_t>(dimension));
  }
  return shape;
}

Tensor parse_npy(const std::vector<uint8_t>& file, const TensorSpec& spec) {
  const std::string member_name = spec.archive_name + ".npy";
  constexpr uint8_t kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (file.size() < 10 ||
      !std::equal(std::begin(kMagic), std::end(kMagic), file.begin())) {
    throw std::runtime_error("NPZ member '" + member_name +
                             "' is not a valid NPY file.");
  }

  const uint8_t major = file[6];
  size_t header_start = 0;
  size_t header_length = 0;
  if (major == 1) {
    header_start = 10;
    header_length = read_little_u16(file.data() + 8);
  } else if (major == 2 || major == 3) {
    if (file.size() < 12) {
      throw std::runtime_error("NPY member '" + member_name+
                               "' has a truncated header.");
    }
    header_start = 12;
    header_length = read_little_u32(file.data() + 8);
  } else {
    throw std::runtime_error("Unsupported NPY version in member '" +
                             member_name + "'.");
  }
  if (header_length > file.size() - header_start) {
    throw std::runtime_error("NPY member '" + member_name+
                             "' has a truncated header.");
  }

  const std::string header(
      reinterpret_cast<const char*>(file.data() + header_start),
      header_length);
  static const std::regex descr_pattern(
      R"npy(['"]descr['"]\s*:\s*['"]([^'"]+)['"])npy");
  static const std::regex fortran_pattern(
      R"npy(['"]fortran_order['"]\s*:\s*(True|False))npy");
  static const std::regex shape_pattern(
      R"npy(['"]shape['"]\s*:\s*\(([^)]*)\))npy");

  std::smatch match;
  if (!std::regex_search(header, match, descr_pattern)) {
    throw std::runtime_error("NPY member '" + member_name+
                             "' has no datatype descriptor.");
  }
  std::string descriptor = match[1].str();
  const std::string expected = expected_numpy_descr(spec.type);
  const std::string native = "=" + expected.substr(1);
  if (descriptor != expected && descriptor != native) {
    throw std::runtime_error("Input '" + spec.archive_name + "' has dtype '" +
                             descriptor + "'; expected '" + expected + "'.");
  }

  if (!std::regex_search(header, match, fortran_pattern) ||
      match[1].str() != "False") {
    throw std::runtime_error("Input '" + spec.archive_name+
                             "' must use C-contiguous array order.");
  }
  if (!std::regex_search(header, match, shape_pattern)) {
    throw std::runtime_error("NPY member '" + member_name+
                             "' has no valid shape.");
  }
  std::vector<int64_t> shape =
      parse_numpy_shape(match[1].str(), member_name);

  const size_t data_start = header_start + header_length;
  const size_t element_count = checked_element_count(shape);
  if (element_count > std::numeric_limits<size_t>::max() / scalar_size(spec.type)) {
    throw std::runtime_error("NPY member '" + member_name + "' is too large.");
  }
  const size_t expected_bytes = element_count * scalar_size(spec.type);
  if (file.size() - data_start != expected_bytes) {
    throw std::runtime_error("NPY member '" + member_name+
                             "' byte count does not match its shape.");
  }

  Tensor tensor = tensor_from_bytes(spec.type, std::move(shape),
                                    file.data() + data_start, expected_bytes);
  validate_fixed_contract(spec.archive_name, tensor, spec);
  return tensor;
}

int64_t sum_int32(const Tensor& tensor) {
  int64_t total = 0;
  for (const int32_t value : values<int32_t>(tensor)) {
    if ((value > 0 && total > std::numeric_limits<int64_t>::max() - value) ||
        (value < 0 && total < std::numeric_limits<int64_t>::min() - value)) {
      throw std::runtime_error("Integer overflow while summing a count tensor.");
    }
    total += value;
  }
  return total;
}

void require_equal(size_t actual, size_t expected,
                   const std::string& description) {
  if (actual != expected) {
    throw std::runtime_error(description + ": got " + std::to_string(actual) +
                             ", expected " + std::to_string(expected) + ".");
  }
}

void require_equal(int64_t actual, int64_t expected,
                   const std::string& description) {
  if (actual != expected) {
    throw std::runtime_error(description + ": got " + std::to_string(actual) +
                             ", expected " + std::to_string(expected) + ".");
  }
}

void validate_input_relationships(const TensorMap& inputs,
                                  bool use_flash_match) {
  const Tensor& coordinates = at(inputs, "coordinates");
  const Tensor& features = at(inputs, "features");
  const Tensor& sources = at(inputs, "sources");
  const Tensor& counts = at(inputs, "counts");
  const Tensor& meta = at(inputs, "meta");
  const Tensor& run_info = at(inputs, "run_info");

  const auto& count_values = values<int32_t>(counts);
  if (std::any_of(count_values.begin(), count_values.end(),
                  [](int32_t value) { return value < 0; })) {
    throw std::runtime_error("Input 'counts' cannot contain negative values.");
  }

  const size_t voxel_count = rows(coordinates);
  const size_t event_count = rows(counts);
  if (rows(features) != voxel_count || rows(sources) != voxel_count) {
    throw std::runtime_error(
        "coordinates, features, and sources must have the same row count.");
  }
  require_equal(sum_int32(counts), static_cast<int64_t>(voxel_count),
                "sum(counts)");
  if (rows(meta) != event_count || rows(run_info) != event_count) {
    throw std::runtime_error(
        "counts, meta, and run_info must describe the same events.");
  }

  const auto& coordinate_values = values<int32_t>(coordinates);
  size_t voxel = 0;
  for (size_t event = 0; event < event_count; ++event) {
    if (event > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("Too many request-local event IDs.");
    }
    for (int32_t index = 0; index < count_values[event]; ++index, ++voxel) {
      if (coordinate_values[voxel * 4] != static_cast<int32_t>(event)) {
        throw std::runtime_error(
            "The first COORDINATES column must contain contiguous request-local "
            "event IDs 0, 1, ... according to counts.");
      }
    }
  }

  if (use_flash_match) {
    const Tensor& flash_data = at(inputs, "flash_data");
    const Tensor& flash_pe = at(inputs, "flash_pe");
    const Tensor& flash_counts = at(inputs, "flash_counts");
    const auto& flash_count_values = values<int32_t>(flash_counts);
    if (rows(flash_pe) != rows(flash_data)) {
      throw std::runtime_error(
          "flash_data and flash_pe must have the same row count.");
    }
    if (flash_counts.shape != counts.shape) {
      throw std::runtime_error(
          "flash_counts must contain one value per event.");
    }
    if (std::any_of(flash_count_values.begin(), flash_count_values.end(),
                    [](int32_t value) { return value < 0; })) {
      throw std::runtime_error(
          "Input 'flash_counts' cannot contain negative values.");
    }
    require_equal(sum_int32(flash_counts),
                  static_cast<int64_t>(rows(flash_data)),
                  "sum(flash_counts)");
  }
}

struct LoadedInputs {
  TensorMap tensors;
  bool use_flash_match;
};

LoadedInputs load_inputs(const fs::path& input_path, bool skip_flash_match) {
  if (!fs::is_regular_file(input_path)) {
    throw std::runtime_error("Input file not found: " + input_path.string());
  }

  NpzArchive archive(input_path);
  std::vector<std::string> missing;
  for (const auto& spec : kInputSpecs) {
    if (!archive.contains(spec.archive_name + ".npy")) {
      missing.push_back(spec.archive_name);
    }
  }
  if (!missing.empty()) {
    throw std::runtime_error("Input NPZ is missing required arrays: " +
                             join(missing, ", "));
  }

  size_t present_flash = 0;
  for (const auto& spec : kFlashInputSpecs) {
    if (archive.contains(spec.archive_name + ".npy")) {
      ++present_flash;
    }
  }
  if (!skip_flash_match && present_flash != 0 &&
      present_flash != kFlashInputSpecs.size()) {
    std::vector<std::string> missing_flash;
    for (const auto& spec : kFlashInputSpecs) {
      if (!archive.contains(spec.archive_name + ".npy")) {
        missing_flash.push_back(spec.archive_name);
      }
    }
    throw std::runtime_error(
        "The optical inputs are an all-or-none group. Missing: " +
        join(missing_flash, ", "));
  }

  const bool use_flash_match =
      !skip_flash_match && present_flash == kFlashInputSpecs.size();
  TensorMap tensors;
  auto load_group = [&archive, &tensors](const std::vector<TensorSpec>& specs) {
    for (const auto& spec : specs) {
      Tensor tensor = parse_npy(archive.read(spec.archive_name + ".npy"), spec);
      tensors.emplace(spec.archive_name, std::move(tensor));
    }
  };
  load_group(kInputSpecs);
  if (use_flash_match) {
    load_group(kFlashInputSpecs);
  }

  validate_input_relationships(tensors, use_flash_match);
  return LoadedInputs{std::move(tensors), use_flash_match};
}

void check_triton(const tc::Error& error, const std::string& context) {
  if (!error.IsOk()) {
    throw std::runtime_error(context + ": " + error.Message());
  }
}

struct RequestObjects {
  std::vector<std::shared_ptr<tc::InferInput>> input_owners;
  std::vector<tc::InferInput*> inputs;
  std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_owners;
  std::vector<const tc::InferRequestedOutput*> outputs;
};

RequestObjects build_request(const TensorMap& tensors, bool use_flash_match) {
  RequestObjects request;
  static const uint8_t empty_sentinel = 0;

  auto add_inputs = [&request, &tensors](
                        const std::vector<TensorSpec>& specs) {
    for (const auto& spec : specs) {
      const Tensor& tensor = at(tensors, spec.archive_name);
      tc::InferInput* raw_input = nullptr;
      check_triton(tc::InferInput::Create(&raw_input, spec.triton_name,
                                          tensor.shape, scalar_name(spec.type)),
                   "Cannot create Triton input '" + spec.triton_name + "'");
      std::shared_ptr<tc::InferInput> input(raw_input);
      const uint8_t* data = tensor.raw_data();
      if (data == nullptr) {
        data = &empty_sentinel;
      }
      check_triton(input->AppendRaw(data, tensor.byte_size()),
                   "Cannot attach data to Triton input '" + spec.triton_name+
                       "'");
      request.inputs.push_back(input.get());
      request.input_owners.push_back(std::move(input));
    }
  };
  add_inputs(kInputSpecs);
  if (use_flash_match) {
    add_inputs(kFlashInputSpecs);
  }

  for (const auto& spec : kOutputSpecs) {
    tc::InferRequestedOutput* raw_output = nullptr;
    check_triton(tc::InferRequestedOutput::Create(&raw_output, spec.triton_name),
                 "Cannot request Triton output '" + spec.triton_name + "'");
    std::shared_ptr<tc::InferRequestedOutput> output(raw_output);
    request.outputs.push_back(output.get());
    request.output_owners.push_back(std::move(output));
  }
  return request;
}

TensorMap collect_outputs(tc::InferResult& result) {
  TensorMap outputs;
  for (const auto& spec : kOutputSpecs) {
    std::vector<int64_t> shape;
    std::string datatype;
    const uint8_t* data = nullptr;
    size_t byte_count = 0;
    check_triton(result.Shape(spec.triton_name, &shape),
                 "Response is missing shape for '" + spec.triton_name + "'");
    check_triton(result.Datatype(spec.triton_name, &datatype),
                 "Response is missing datatype for '" + spec.triton_name+
                     "'");
    if (datatype != scalar_name(spec.type)) {
      throw std::runtime_error("Output '" + spec.triton_name +
                               "' has datatype " + datatype + "; expected " +
                               scalar_name(spec.type) + ".");
    }
    check_triton(result.RawData(spec.triton_name, &data, &byte_count),
                 "Response is missing data for '" + spec.triton_name + "'");
    if (byte_count != 0 && data == nullptr) {
      throw std::runtime_error("Output '" + spec.triton_name+
                               "' has a null data buffer.");
    }
    Tensor tensor =
        tensor_from_bytes(spec.type, std::move(shape), data, byte_count);
    validate_fixed_contract(spec.triton_name, tensor, spec);
    outputs.emplace(spec.triton_name, std::move(tensor));
  }
  return outputs;
}

template <typename Value>
bool equal_vectors(const Tensor& left, const Tensor& right) {
  return values<Value>(left) == values<Value>(right);
}

void validate_outputs(const TensorMap& outputs, const TensorMap& inputs,
                      bool use_flash_match) {
  const size_t event_count = rows(at(inputs, "counts"));
  const size_t input_voxel_count = rows(at(inputs, "coordinates"));
  const size_t deghosted_count = rows(at(outputs, "SEGMENTATION"));
  const size_t particle_count = rows(at(outputs, "PARTICLES"));
  const size_t membership_count = rows(at(outputs, "PARTICLE_VOXELS"));
  const size_t interaction_count = rows(at(outputs, "INTERACTIONS"));
  const size_t flash_association_count =
      rows(at(outputs, "INTERACTION_FLASHES"));

  const Tensor& event_id = at(outputs, "EVENT_ID");
  if (event_id.shape != std::vector<int64_t>{static_cast<int64_t>(event_count),
                                             3}) {
    throw std::runtime_error("EVENT_ID has an invalid shape.");
  }
  if (!equal_vectors<int64_t>(event_id, at(inputs, "run_info"))) {
    throw std::runtime_error("EVENT_ID does not match the request RUN_INFO.");
  }

  require_equal(rows(at(outputs, "GHOST_PRED")), input_voxel_count,
                "GHOST_PRED rows");
  const Tensor& ghost_counts = at(outputs, "GHOST_COUNTS");
  if (ghost_counts.shape !=
      std::vector<int64_t>{static_cast<int64_t>(event_count)}) {
    throw std::runtime_error(
        "GHOST_COUNTS must contain one value per event.");
  }
  if (!equal_vectors<int32_t>(ghost_counts, at(inputs, "counts"))) {
    throw std::runtime_error(
        "GHOST_COUNTS does not match the request COUNTS.");
  }

  const Tensor& segmentation_counts = at(outputs, "SEGMENTATION_COUNTS");
  if (segmentation_counts.shape !=
      std::vector<int64_t>{static_cast<int64_t>(event_count)}) {
    throw std::runtime_error(
        "SEGMENTATION_COUNTS must contain one value per event.");
  }
  require_equal(sum_int32(segmentation_counts),
                static_cast<int64_t>(deghosted_count),
                "sum(SEGMENTATION_COUNTS)");
  require_equal(rows(at(outputs, "ORIG_INDEX")), deghosted_count,
                "ORIG_INDEX rows");
  require_equal(rows(at(outputs, "POINTS")), deghosted_count, "POINTS rows");
  require_equal(rows(at(outputs, "DEPOSITIONS")), deghosted_count,
                "DEPOSITIONS rows");
  require_equal(rows(at(outputs, "DEGHOSTED_SOURCES")), deghosted_count,
                "DEGHOSTED_SOURCES rows");

  const auto& original_indices = values<int64_t>(at(outputs, "ORIG_INDEX"));
  const auto& ghost_prediction = values<int64_t>(at(outputs, "GHOST_PRED"));
  for (const int64_t index : original_indices) {
    if (index < 0 || static_cast<uint64_t>(index) >= input_voxel_count) {
      throw std::runtime_error(
          "ORIG_INDEX contains values outside the request row range.");
    }
    if (ghost_prediction[static_cast<size_t>(index)] != 0) {
      throw std::runtime_error(
          "ORIG_INDEX points to one or more predicted ghost voxels.");
    }
  }

  const Tensor& particle_counts = at(outputs, "PARTICLE_COUNTS");
  if (particle_counts.shape !=
      std::vector<int64_t>{static_cast<int64_t>(event_count)}) {
    throw std::runtime_error(
        "PARTICLE_COUNTS must contain one value per event.");
  }
  require_equal(sum_int32(particle_counts), static_cast<int64_t>(particle_count),
                "sum(PARTICLE_COUNTS)");
  const Tensor& particle_voxel_counts =
      at(outputs, "PARTICLE_VOXEL_COUNTS");
  require_equal(rows(particle_voxel_counts), particle_count,
                "PARTICLE_VOXEL_COUNTS rows");
  require_equal(sum_int32(particle_voxel_counts),
                static_cast<int64_t>(membership_count),
                "sum(PARTICLE_VOXEL_COUNTS)");

  const Tensor& interaction_counts = at(outputs, "INTERACTION_COUNTS");
  if (interaction_counts.shape !=
      std::vector<int64_t>{static_cast<int64_t>(event_count)}) {
    throw std::runtime_error(
        "INTERACTION_COUNTS must contain one value per event.");
  }
  require_equal(sum_int32(interaction_counts),
                static_cast<int64_t>(interaction_count),
                "sum(INTERACTION_COUNTS)");
  const Tensor& interaction_flash_counts =
      at(outputs, "INTERACTION_FLASH_COUNTS");
  require_equal(rows(interaction_flash_counts), interaction_count,
                "INTERACTION_FLASH_COUNTS rows");
  require_equal(sum_int32(interaction_flash_counts),
                static_cast<int64_t>(flash_association_count),
                "sum(INTERACTION_FLASH_COUNTS)");

  const Tensor& flash_match_ran = at(outputs, "FLASH_MATCH_RAN");
  if (flash_match_ran.shape !=
      std::vector<int64_t>{static_cast<int64_t>(event_count)}) {
    throw std::runtime_error(
        "FLASH_MATCH_RAN must contain one value per event.");
  }
  const int32_t expected_flag = use_flash_match ? 1 : 0;
  if (std::any_of(values<int32_t>(flash_match_ran).begin(),
                  values<int32_t>(flash_match_ran).end(),
                  [expected_flag](int32_t value) {
                    return value != expected_flag;
                  })) {
    throw std::runtime_error(
        "FLASH_MATCH_RAN is inconsistent with the optical inputs sent.");
  }
}

std::string shape_string(const Tensor& tensor) {
  std::ostringstream stream;
  stream << '(';
  for (size_t index = 0; index < tensor.shape.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << tensor.shape[index];
  }
  if (tensor.shape.size() == 1) {
    stream << ',';
  }
  stream << ')';
  return stream.str();
}

void print_value(const Tensor& tensor, size_t index, std::ostream& output) {
  std::visit(
      [index, &output](const auto& tensor_values) {
        output << tensor_values.at(index);
      },
      tensor.storage);
}

std::string preview(const Tensor& tensor, size_t requested_rows,
                    size_t maximum_columns = 8) {
  if (requested_rows == 0 || tensor.element_count() == 0) {
    return "";
  }

  std::ostringstream stream;
  stream << std::setprecision(6);
  if (tensor.shape.size() == 1) {
    const size_t count = std::min(requested_rows, tensor.element_count());
    stream << '[';
    for (size_t index = 0; index < count; ++index) {
      if (index != 0) {
        stream << ' ';
      }
      print_value(tensor, index, stream);
    }
    stream << ']';
    return stream.str();
  }

  const size_t row_count = std::min(requested_rows, rows(tensor));
  const size_t columns = static_cast<size_t>(tensor.shape.at(1));
  const size_t shown_columns = std::min(columns, maximum_columns);
  stream << '[';
  for (size_t row = 0; row < row_count; ++row) {
    if (row != 0) {
      stream << '\n' << ' ';
    }
    stream << '[';
    for (size_t column = 0; column < shown_columns; ++column) {
      if (column != 0) {
        stream << ' ';
      }
      print_value(tensor, row * columns + column, stream);
    }
    stream << ']';
  }
  stream << ']';
  return stream.str();
}

void print_summary(const TensorMap& outputs, double elapsed_seconds,
                   int preview_rows) {
  std::cout << "\nInference result\n";
  std::cout << "  Event IDs:           "
            << preview(at(outputs, "EVENT_ID"), rows(at(outputs, "EVENT_ID")), 3)
            << '\n';
  std::cout << "  Input voxels:        "
            << sum_int32(at(outputs, "GHOST_COUNTS")) << '\n';
  std::cout << "  Deghosted voxels:    " << rows(at(outputs, "SEGMENTATION"))
            << '\n';
  std::cout << "  Particles:           " << rows(at(outputs, "PARTICLES"))
            << '\n';
  std::cout << "  Interactions:        " << rows(at(outputs, "INTERACTIONS"))
            << '\n';
  std::cout << "  FlashMatch ran:      "
            << preview(at(outputs, "FLASH_MATCH_RAN"),
                       rows(at(outputs, "FLASH_MATCH_RAN")))
            << '\n';
  std::cout << "  Flash associations: "
            << rows(at(outputs, "INTERACTION_FLASHES")) << '\n';
  std::cout << "  Inference time:      " << std::fixed << std::setprecision(3)
            << elapsed_seconds << " s\n";

  std::cout << "\nReturned arrays\n";
  for (const auto& spec : kOutputSpecs) {
    const Tensor& tensor = at(outputs, spec.triton_name);
    std::cout << "  " << std::left << std::setw(27) << spec.triton_name
              << " shape=" << std::setw(14) << shape_string(tensor)
              << " dtype=" << scalar_name(tensor.type) << '\n';
    const std::string sample =
        preview(tensor, static_cast<size_t>(preview_rows));
    if (!sample.empty()) {
      std::cout << "    first "
                << std::min(static_cast<size_t>(preview_rows), rows(tensor))
                << ": " << sample << '\n';
    }
  }
  std::cout << std::right << std::defaultfloat;
}

const H5::PredType& hdf5_file_type(ScalarType type) {
  switch (type) {
    case ScalarType::Int32:
      return H5::PredType::STD_I32LE;
    case ScalarType::Int64:
      return H5::PredType::STD_I64LE;
    case ScalarType::Float32:
      return H5::PredType::IEEE_F32LE;
  }
  throw std::logic_error("Unknown scalar type.");
}

const H5::PredType& hdf5_memory_type(ScalarType type) {
  switch (type) {
    case ScalarType::Int32:
      return H5::PredType::NATIVE_INT32;
    case ScalarType::Int64:
      return H5::PredType::NATIVE_INT64;
    case ScalarType::Float32:
      return H5::PredType::NATIVE_FLOAT;
  }
  throw std::logic_error("Unknown scalar type.");
}

void save_outputs(const fs::path& output_path, const TensorMap& outputs,
                  bool overwrite) {
  if (fs::exists(output_path) && !overwrite) {
    throw std::runtime_error("Output file already exists: " +
                             output_path.string() +
                             ". Use --overwrite to replace it.");
  }
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path());
  }

  H5::Exception::dontPrint();
  H5::H5File file(output_path.string(), H5F_ACC_TRUNC);
  for (const auto& spec : kOutputSpecs) {
    const Tensor& tensor = at(outputs, spec.triton_name);
    std::vector<hsize_t> dimensions;
    dimensions.reserve(tensor.shape.size());
    for (const int64_t value : tensor.shape) {
      dimensions.push_back(static_cast<hsize_t>(value));
    }
    H5::DataSpace space(static_cast<int>(dimensions.size()), dimensions.data());
    H5::DSetCreatPropList properties;
    if (tensor.element_count() != 0) {
      std::vector<hsize_t> chunks = dimensions;
      chunks[0] = std::min<hsize_t>(chunks[0], tensor.shape.size() == 1 ? 65536
                                                                       : 4096);
      properties.setChunk(static_cast<int>(chunks.size()), chunks.data());
      properties.setShuffle();
      properties.setDeflate(4);
    }
    H5::DataSet dataset = file.createDataSet(
        spec.triton_name, hdf5_file_type(tensor.type), space, properties);
    if (tensor.element_count() != 0) {
      dataset.write(tensor.raw_data(), hdf5_memory_type(tensor.type));
    }
  }
}

class CurlGlobal {
 public:
  CurlGlobal() {
    const CURLcode code = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (code != CURLE_OK) {
      throw std::runtime_error("Cannot initialize libcurl: " +
                               std::string(curl_easy_strerror(code)));
    }
  }
  ~CurlGlobal() { curl_global_cleanup(); }
  CurlGlobal(const CurlGlobal&) = delete;
  CurlGlobal& operator=(const CurlGlobal&) = delete;
};

size_t discard_curl_body(char*, size_t size, size_t count, void*) {
  return size * count;
}

long milliseconds(double seconds) {
  const double value = std::ceil(seconds * 1000.0);
  if (!std::isfinite(value) || value > std::numeric_limits<long>::max()) {
    throw std::runtime_error("HTTP timeout is too large.");
  }
  return static_cast<long>(value);
}

uint64_t microseconds(double seconds) {
  const long double value =
      std::ceil(static_cast<long double>(seconds) * 1000000.0L);
  if (!std::isfinite(static_cast<double>(value)) ||
      value > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("Inference timeout is too large.");
  }
  return static_cast<uint64_t>(value);
}

std::string http_base_url(std::string url) {
  if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
    url = "http://" + url;
  }
  while (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  return url;
}

std::string curl_escape(CURL* curl, const std::string& value) {
  char* escaped =
      curl_easy_escape(curl, value.c_str(), static_cast<int>(value.size()));
  if (escaped == nullptr) {
    throw std::runtime_error("Cannot URL-encode a Triton model identifier.");
  }
  std::string result(escaped);
  curl_free(escaped);
  return result;
}

void require_http_ready(const std::string& url, const std::string& description,
                        double connection_timeout, double network_timeout) {
  CURL* raw_curl = curl_easy_init();
  if (raw_curl == nullptr) {
    throw std::runtime_error("Cannot create an HTTP readiness request.");
  }
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(raw_curl,
                                                          &curl_easy_cleanup);
  char error_buffer[CURL_ERROR_SIZE] = {};
  curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                   milliseconds(connection_timeout));
  curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, milliseconds(network_timeout));
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, discard_curl_body);
  curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, error_buffer);
  curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);

  const CURLcode code = curl_easy_perform(curl.get());
  if (code != CURLE_OK) {
    const std::string detail = error_buffer[0] != '\0'
                                   ? error_buffer
                                   : curl_easy_strerror(code);
    throw std::runtime_error(description + " check failed: " + detail);
  }
  long status = 0;
  curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
  if (status != 200) {
    throw std::runtime_error(description + " is not ready (HTTP " +
                             std::to_string(status) + ").");
  }
}

void check_readiness(const Arguments& args) {
  const std::string base = http_base_url(args.url);
  require_http_ready(base + "/v2/health/live", "Triton server liveness",
                     args.connection_timeout, args.network_timeout);
  require_http_ready(base + "/v2/health/ready", "Triton server readiness",
                     args.connection_timeout, args.network_timeout);

  CURL* raw_curl = curl_easy_init();
  if (raw_curl == nullptr) {
    throw std::runtime_error("Cannot create a URL encoder.");
  }
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(raw_curl,
                                                          &curl_easy_cleanup);
  const std::string model = curl_escape(curl.get(), args.model_name);
  const std::string version = curl_escape(curl.get(), args.model_version);
  require_http_ready(base + "/v2/models/" + model + "/versions/" + version +
                         "/ready",
                     "Model '" + args.model_name + "', version '" +
                         args.model_version + "'",
                     args.connection_timeout, args.network_timeout);
}

double parse_positive_double(const std::string& text,
                             const std::string& option) {
  size_t parsed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(option + " requires a numeric value.");
  }
  if (parsed != text.size() || !std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(option + " must be greater than zero.");
  }
  return value;
}

int parse_nonnegative_int(const std::string& text,
                          const std::string& option) {
  size_t parsed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(option + " requires an integer value.");
  }
  if (parsed != text.size() || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(option + " cannot be negative.");
  }
  return static_cast<int>(value);
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " INPUT.npz [options]\n\n"
      << "Query the ICARUS SPINE full-chain model through Triton HTTP and\n"
      << "save all returned tensors in an HDF5 file.\n\n"
      << "Options:\n"
      << "  -o, --output PATH          Output .h5/.hdf5 path\n"
      << "      --url URL              Triton endpoint (default: "
      << kDefaultUrl << ")\n"
      << "      --model-name NAME      Model name (default: "
      << kDefaultModelName << ")\n"
      << "      --model-version VER    Model version (default: "
      << kDefaultModelVersion << ")\n"
      << "      --skip-flash-match     Omit all optical inputs\n"
      << "      --connection-timeout S HTTP connection timeout (default: 10)\n"
      << "      --network-timeout S    End-to-end timeout (default: 3600)\n"
      << "      --preview N            Preview N rows (default: 2)\n"
      << "      --overwrite            Replace an existing output file\n"
      << "      --verbose              Enable Triton HTTP client logging\n"
      << "  -h, --help                 Show this help\n";
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  std::vector<std::string> positional;

  auto next_value = [argc, argv](int& index, const std::string& option) {
    if (index + 1 >= argc) {
      throw std::runtime_error(option + " requires a value.");
    }
    return std::string(argv[++index]);
  };

  for (int index = 1; index < argc; ++index) {
    const std::string token = argv[index];
    if (token == "-h" || token == "--help") {
      args.help = true;
    } else if (token == "-o" || token == "--output") {
      args.output = fs::path(next_value(index, token));
    } else if (token == "--url") {
      args.url = next_value(index, token);
    } else if (token == "--model-name") {
      args.model_name = next_value(index, token);
    } else if (token == "--model-version") {
      args.model_version = next_value(index, token);
    } else if (token == "--skip-flash-match") {
      args.skip_flash_match = true;
    } else if (token == "--connection-timeout") {
      args.connection_timeout =
          parse_positive_double(next_value(index, token), token);
    } else if (token == "--network-timeout") {
      args.network_timeout =
          parse_positive_double(next_value(index, token), token);
    } else if (token == "--preview") {
      args.preview = parse_nonnegative_int(next_value(index, token), token);
    } else if (token == "--overwrite") {
      args.overwrite = true;
    } else if (token == "--verbose") {
      args.verbose = true;
    } else if (!token.empty() && token.front() == '-') {
      throw std::runtime_error("Unknown option: " + token);
    } else {
      positional.push_back(token);
    }
  }

  if (!args.help) {
    if (positional.empty()) {
      throw std::runtime_error("Missing input NPZ path. Use --help for usage.");
    }
    if (positional.size() != 1) {
      throw std::runtime_error("Expected exactly one input NPZ path.");
    }
    args.input = positional.front();
    if (args.url.empty() || args.model_name.empty() || args.model_version.empty()) {
      throw std::runtime_error(
          "URL, model name, and model version cannot be empty.");
    }
  }
  return args;
}

fs::path default_output_path(const fs::path& input_path) {
  if (lower(input_path.extension().string()) == ".npz") {
    return input_path.parent_path() /
           fs::path(input_path.stem().string() + "_triton.h5");
  }
  return fs::path(input_path.string() + "_triton.h5");
}

fs::path normalize_output_path(fs::path path) {
  if (path.extension().empty()) {
    path = fs::path(path.string() + ".h5");
  }
  const std::string suffix = lower(path.extension().string());
  if (suffix != ".h5" && suffix != ".hdf5") {
    throw std::runtime_error(
        "The output file must use the .h5 or .hdf5 extension.");
  }
  return path;
}

int run(int argc, char** argv) {
  const Arguments args = parse_arguments(argc, argv);
  if (args.help) {
    print_usage(argv[0]);
    return 0;
  }

  const fs::path input_path = fs::absolute(args.input).lexically_normal();
  const fs::path requested_output =
      args.output.has_value() ? *args.output : default_output_path(input_path);
  const fs::path output_path =
      fs::absolute(normalize_output_path(requested_output)).lexically_normal();
  if (input_path == output_path) {
    throw std::runtime_error(
        "The output path must be different from the input path.");
  }

  LoadedInputs loaded = load_inputs(input_path, args.skip_flash_match);

  CurlGlobal curl_global;

  std::unique_ptr<tc::InferenceServerHttpClient> client;
  check_triton(tc::InferenceServerHttpClient::Create(&client, args.url,
                                                      args.verbose),
               "Cannot create Triton HTTP client");

  // Check readiness before preparing and transferring the request.
  check_readiness(args);
  std::cout << "Connected: " << args.url << '\n';
  std::cout << "Model ready: " << args.model_name << ", version "
            << args.model_version << '\n';
  std::cout << "Input: " << input_path << '\n';
  std::cout << "Optical inputs: "
            << (loaded.use_flash_match ? "enabled" : "omitted") << '\n';

  RequestObjects request =
      build_request(loaded.tensors, loaded.use_flash_match);

  tc::InferOptions options(args.model_name);
  options.model_version_ = args.model_version;
  options.client_timeout_ = microseconds(args.network_timeout);

  tc::InferResult* raw_result = nullptr;
  const auto start = std::chrono::steady_clock::now();
  check_triton(client->Infer(&raw_result, options, request.inputs,
                             request.outputs),
               "Triton inference request failed");
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::unique_ptr<tc::InferResult> result(raw_result);

  // The result owns the response buffer; the connection can now be released.
  client.reset();

  TensorMap outputs = collect_outputs(*result);
  validate_outputs(outputs, loaded.tensors, loaded.use_flash_match);
  print_summary(outputs, elapsed, args.preview);
  save_outputs(output_path, outputs, args.overwrite);

  std::cout << "\nSaved: " << output_path << '\n';
  std::cout << "ICARUS SPINE Triton output contract: PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const H5::Exception& error) {
    std::cerr << "ERROR: HDF5: " << error.getDetailMsg() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
  }
  return 1;
}
