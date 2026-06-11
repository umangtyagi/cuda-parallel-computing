#include <cfloat>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

// ------------------------------------ RAY CASTING
// ------------------------------------ Structure to represent a 3D point
struct Point3D {
  float x, y, z;

  Point3D operator+(const Point3D &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }

  Point3D operator-(const Point3D &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }

  Point3D operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }
};

// Structure to represent a voxel grid
struct VoxelGrid {
  int sizeX, sizeY, sizeZ; // Number of voxels along each dimension
  float voxelSize;         // Size of each voxel
  std::vector<bool> grid; // True indicates occupied, False indicates free space

  VoxelGrid(int x, int y, int z, float size)
      : sizeX(x), sizeY(y), sizeZ(z), voxelSize(size), grid(x * y * z, false) {}

  // Convert voxel coordinates to grid vector index
  int getIndex(int x, int y, int z) const {
    if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
      return x + y * sizeX + z * sizeX * sizeY;
    }
    return -1; // Invalid index
  }

  // Check if a voxel is occupied
  bool isOccupied(int x, int y, int z) const {
    int index = getIndex(x, y, z);
    return index != -1 && grid[index];
  }

  // Set a voxel to occupied
  void setOccupied(int x, int y, int z) {
    int index = getIndex(x, y, z);
    if (index != -1) {
      grid[index] = true;
    }
  }

  // Convert a Point3D to voxel grid coordinates
  std::tuple<int, int, int> pointToGridIndex(const Point3D &point) const {
    int x = static_cast<int>(point.x / voxelSize);
    int y = static_cast<int>(point.y / voxelSize);
    int z = static_cast<int>(point.z / voxelSize);
    return {x, y, z};
  }

  // Convert voxel grid coordinates to a Point3D (center of the voxel)
  Point3D gridIndexToPoint(int x, int y, int z) const {
    float px = (x + 0.5f) * voxelSize;
    float py = (y + 0.5f) * voxelSize;
    float pz = (z + 0.5f) * voxelSize;
    return {px, py, pz};
  }

  // Insert a point cloud into the voxel grid
  void insertPointCloud(const std::vector<Point3D> &points) {
    for (const auto &point : points) {
      // Compute voxel coordinates from the point
      std::tuple<int, int, int> voxelIndex = pointToGridIndex(point);
      int x = std::get<0>(voxelIndex);
      int y = std::get<1>(voxelIndex);
      int z = std::get<2>(voxelIndex);
      // Check if the point is within the valid range of the voxel grid
      if (point.x < 0 || point.x >= voxelSize * sizeX || point.y < 0 ||
          point.y >= voxelSize * sizeY || point.z < 0 ||
          point.z >= voxelSize * sizeZ) {
        std::cerr << "Point (" << point.x << ", " << point.y << ", " << point.z
                  << ") is out of bounds. Skipping this point.\n";
        continue; // Skip points that are out of bounds
      }
      // Set the corresponding voxel as occupied
      setOccupied(x, y, z);
    }
  }

  // Extract the point cloud
  std::vector<Point3D> extractPointCloud() const {
    std::vector<Point3D> pointCloud;
    // Iterate through all voxels in the grid
    for (int z = 0; z < sizeZ; ++z) {
      for (int y = 0; y < sizeY; ++y) {
        for (int x = 0; x < sizeX; ++x) {
          // Check if the voxel is occupied
          if (isOccupied(x, y, z)) {
            // Convert grid coordinates to the corresponding Point3D (center of
            // the voxel)
            Point3D point = gridIndexToPoint(x, y, z);
            pointCloud.push_back(point);
          }
        }
      }
    }
    return pointCloud;
  }

  // Perform ray casting in the voxel grid
  bool rayCasting(const Point3D &start, const Point3D &direction,
                  Point3D &endPoint, float maxDistance) {
    // Normalize the direction vector
    Point3D normalizedDirection =
        direction * (1.0f / std::sqrt(direction.x * direction.x +
                                      direction.y * direction.y +
                                      direction.z * direction.z));

    // Current position along the ray
    Point3D current = start;
    float stepSize = voxelSize / 2.0f; // Step size for ray traversal
    float distance = 0.0f;             // Total traveled distance

    // Ray stop until hits an occupied voxel or exceeds the maxDistance
    while (distance < maxDistance) {
      // Convert the current position to voxel grid coordinates
      std::tuple<int, int, int> voxelIndex = pointToGridIndex(current);
      int x = std::get<0>(voxelIndex);
      int y = std::get<1>(voxelIndex);
      int z = std::get<2>(voxelIndex);

      // Check if the current voxel is occupied
      if (isOccupied(x, y, z)) {
        // std::cout << "Hit at voxel (" << x << ", " << y << ", " << z << ")"
        // << std::endl;
        endPoint = gridIndexToPoint(x, y, z);
        return true; // Hit an occupied voxel
      }

      // Move along the ray
      current = current + normalizedDirection * stepSize;
      distance += stepSize;
    }

    // std::cout << "No hit within max distance." << std::endl;
    return false; // No hit within maxDistance
  }
};

// ------------------------------------ CHAMFER DISTANCE
// ------------------------------------ Function to compute the Euclidean
// distance between two 3D points
float euclidean_distance(const Point3D &p1, const Point3D &p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                   (p1.y - p2.y) * (p1.y - p2.y) +
                   (p1.z - p2.z) * (p1.z - p2.z));
}

// Brute-force implementation of Chamfer Distance
float chamfer_distance(const std::vector<Point3D> &cloud1,
                       const std::vector<Point3D> &cloud2) {
  float total_distance1 = 0.0f, total_distance2 = 0.0f;

  // Find the shortest distance from each point in cloud1 to cloud2
  for (const auto &p1 : cloud1) {
    float min_distance = std::numeric_limits<float>::max();
    for (const auto &p2 : cloud2) {
      float dist = euclidean_distance(p1, p2);
      if (dist < min_distance)
        min_distance = dist;
    }
    total_distance1 += min_distance;
  }

  // Find the shortest distance from each point in cloud2 to cloud1
  for (const auto &p2 : cloud2) {
    float min_distance = std::numeric_limits<float>::max();
    for (const auto &p1 : cloud1) {
      float dist = euclidean_distance(p2, p1);
      if (dist < min_distance)
        min_distance = dist;
    }
    total_distance2 += min_distance;
  }

  // Return the normalized Chamfer Distance
  return (total_distance1 / cloud1.size()) + (total_distance2 / cloud2.size());
}

// CUDA kernel to compute the nearest distances for each point in cloud1 to
// cloud2
__global__ void chamfer_distance_kernel(const Point3D *cloud1, int n,
                                        const Point3D *cloud2, int m,
                                        float *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  Point3D p1 = cloud1[idx];
  float min_dist = FLT_MAX;

  for (int j = 0; j < m; j++) {
    float dx = p1.x - cloud2[j].x;
    float dy = p1.y - cloud2[j].y;
    float dz = p1.z - cloud2[j].z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist < min_dist)
      min_dist = dist;
  }

  distances[idx] = min_dist;
}

float chamfer_distance_gpu(const std::vector<Point3D> &cloud1,
                           const std::vector<Point3D> &cloud2) {
  int n1 = cloud1.size(), n2 = cloud2.size();

  Point3D *d_c1, *d_c2;
  float *d_dists1, *d_dists2;

  cudaMalloc(&d_c1, n1 * sizeof(Point3D));
  cudaMalloc(&d_c2, n2 * sizeof(Point3D));
  cudaMalloc(&d_dists1, n1 * sizeof(float));
  cudaMalloc(&d_dists2, n2 * sizeof(float));

  cudaMemcpy(d_c1, cloud1.data(), n1 * sizeof(Point3D), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c2, cloud2.data(), n2 * sizeof(Point3D), cudaMemcpyHostToDevice);

  int threads = 256;
  chamfer_distance_kernel<<<(n1 + threads - 1) / threads, threads>>>(
      d_c1, n1, d_c2, n2, d_dists1);
  chamfer_distance_kernel<<<(n2 + threads - 1) / threads, threads>>>(
      d_c2, n2, d_c1, n1, d_dists2);
  cudaDeviceSynchronize();

  std::vector<float> h_dists1(n1), h_dists2(n2);
  cudaMemcpy(h_dists1.data(), d_dists1, n1 * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dists2.data(), d_dists2, n2 * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_c1);
  cudaFree(d_c2);
  cudaFree(d_dists1);
  cudaFree(d_dists2);

  float sum1 = 0, sum2 = 0;
  for (float d : h_dists1)
    sum1 += d;
  for (float d : h_dists2)
    sum2 += d;

  return (sum1 / n1) + (sum2 / n2);
}
// ------------------------------------ PCD FILE
// ------------------------------------
std::vector<Point3D> readPCD(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open PCD file: " + filename);
  }

  std::vector<Point3D> points;
  std::string line;
  bool headerEnded = false;

  while (std::getline(file, line)) {
    // Skip header lines
    if (!headerEnded) {
      if (line.find("DATA ascii") != std::string::npos) {
        headerEnded = true;
      }
      continue;
    }

    // Read point data
    std::istringstream iss(line);
    float x, y, z;
    if (iss >> x >> y >> z) {
      points.push_back({x, y, z});
    }
  }

  file.close();
  return points;
}

void savePCD(const std::string &filename, const std::vector<Point3D> &points) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open PCD file for writing: " +
                             filename);
  }

  // Write PCD header
  file << "# .PCD v0.7 - Point Cloud Data file format\n";
  file << "VERSION 0.7\n";
  file << "FIELDS x y z\n";
  file << "SIZE 4 4 4\n";
  file << "TYPE F F F\n";
  file << "COUNT 1 1 1\n";
  file << "WIDTH " << points.size() << "\n";
  file << "HEIGHT 1\n";
  file << "VIEWPOINT 0 0 0 1 0 0 0\n";
  file << "POINTS " << points.size() << "\n";
  file << "DATA ascii\n";

  // Write points
  for (const auto &point : points) {
    file << point.x << " " << point.y << " " << point.z << "\n";
  }

  file.close();
}

// ------------------------------------ Tuple Hash for Set
// ------------------------------------
struct TupleHash {
  size_t operator()(const std::tuple<int, int, int> &t) const {
    unsigned long long h1 = static_cast<unsigned long long>(std::get<0>(t));
    unsigned long long h2 = static_cast<unsigned long long>(std::get<1>(t));
    unsigned long long h3 = static_cast<unsigned long long>(std::get<2>(t));
    return h1 ^ (h2 << 16) ^ (h3 << 32);
  }
};

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  // Setups
  std::string model_path = "../Models/";

  std::vector<std::string> object_names;
  object_names.push_back("Armadillo");
  //   object_names.push_back("Dragon");

  std::vector<Point3D> view_space;
  std::string view_space_file = "../Viewspace_100.txt";
  std::ifstream fin_vs(view_space_file);
  for (int i = 0; i < 100; i++) {
    float x, y, z;
    fin_vs >> x >> y >> z;
    // Resize normlized view space to 3.0 size sphere and center at object
    // origin
    x *= 3.0;
    y *= 3.0;
    z *= 3.0;
    x += 0.5;
    y += 0.5;
    z += 0.5;
    Point3D view_point = {x, y, z};
    view_space.push_back(view_point);
  }

  // std::vector<int> num_voxels = {64, 128, 256};
  std::vector<int> num_voxels = {64};

  // For each object
  for (std::string object_name : object_names) {
    // For each voxel grid resolution
    for (int num_voxel : num_voxels) {
      // Read object point cloud (all solid points)
      std::string object_file =
          model_path + object_name + "_" + std::to_string(num_voxel) + ".pcd";
      std::vector<Point3D> object_points = readPCD(object_file);
      // As object is normalized to unit cube, voxel size is 1/num_voxel
      float voxel_size = 1.0f / num_voxel;

      // Create a object grid using the object point cloud
      std::cout << "Creating object grid for " << object_name
                << " with resolution " << num_voxel << std::endl;
      VoxelGrid object_grid(num_voxel, num_voxel, num_voxel, voxel_size);
      object_grid.insertPointCloud(object_points);

      // For each view point, do ray casting to get the view space point cloud
      std::vector<std::vector<Point3D>> view_space_points;
      for (int i = 0; i < view_space.size(); i++) {
        Point3D view_point = view_space[i];
        // std::cout << "Ray casting for view point " << std::to_string(i) <<
        // std::endl;
        std::vector<Point3D> view_point_visible_points;
        // traerse all voxels and remain occupied voxels as directions
        for (int x = 0; x < num_voxel; x++) {
          for (int y = 0; y < num_voxel; y++) {
            for (int z = 0; z < num_voxel; z++) {
              // Check if voxel is occupied
              if (!object_grid.isOccupied(x, y, z)) {
                continue;
              }
              // Get voxel center
              Point3D voxel_center = object_grid.gridIndexToPoint(x, y, z);
              // Get ray direction
              Point3D ray_direction = voxel_center - view_point;
              // Get ray origin
              Point3D ray_origin = view_point;
              // Ray cast
              float max_distance = 6.0f;
              Point3D end_point;
              bool hit = object_grid.rayCasting(ray_origin, ray_direction,
                                                end_point, max_distance);
              if (hit) {
                view_point_visible_points.push_back(end_point);
              }
            }
          }
        }
        // Remove duplicate points
        std::unordered_set<std::tuple<int, int, int>, TupleHash> unique_points;
        for (Point3D point : view_point_visible_points) {
          std::tuple<int, int, int> voxelIndex =
              object_grid.pointToGridIndex(point);
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          unique_points.insert(std::make_tuple(x, y, z));
        }
        std::vector<Point3D> view_point_visible_points_unique;
        for (auto it = unique_points.begin(); it != unique_points.end(); it++) {
          std::tuple<int, int, int> voxelIndex = *it;
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          Point3D voxel_center = object_grid.gridIndexToPoint(x, y, z);
          view_point_visible_points_unique.push_back(voxel_center);
        }
        std::cout << "View point " << i << " has "
                  << view_point_visible_points_unique.size()
                  << " visible points" << std::endl;
        view_space_points.push_back(view_point_visible_points_unique);
      }
      // Save view space point cloud
      std::unordered_set<std::tuple<int, int, int>, TupleHash>
          all_visbile_points_set;
      for (int i = 0; i < view_space.size(); i++) {
        std::string view_space_file = "../Output/" + object_name + "_res" +
                                      std::to_string(num_voxel) + "_vp" +
                                      std::to_string(i) + ".pcd";
        savePCD(view_space_file, view_space_points[i]);
        for (Point3D point : view_space_points[i]) {
          std::tuple<int, int, int> voxelIndex =
              object_grid.pointToGridIndex(point);
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          all_visbile_points_set.insert(std::make_tuple(x, y, z));
        }
      }
      // Save all visible points
      std::string all_visbile_points_file = "../Output/" + object_name +
                                            "_res" + std::to_string(num_voxel) +
                                            "_all.pcd";
      std::vector<Point3D> all_visbile_points;
      for (auto it = all_visbile_points_set.begin();
           it != all_visbile_points_set.end(); it++) {
        std::tuple<int, int, int> voxelIndex = *it;
        int x = std::get<0>(voxelIndex);
        int y = std::get<1>(voxelIndex);
        int z = std::get<2>(voxelIndex);
        Point3D voxel_center = object_grid.gridIndexToPoint(x, y, z);
        all_visbile_points.push_back(voxel_center);
      }
      savePCD(all_visbile_points_file, all_visbile_points);
      std::cout << "All visible points number: " << all_visbile_points.size()
                << std::endl;

      // For each view point, select the next best view point
      for (int i = 0; i < view_space.size(); i++) {
        Point3D view_point = view_space[i];
        std::vector<Point3D> view_point_visible_points = view_space_points[i];
        // std::cout << "Selecting next best view point for view point " << i <<
        // std::endl;
        std::unordered_set<std::tuple<int, int, int>, TupleHash>
            reconstructed_points_set;
        for (Point3D point : view_point_visible_points) {
          std::tuple<int, int, int> voxelIndex =
              object_grid.pointToGridIndex(point);
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          reconstructed_points_set.insert(std::make_tuple(x, y, z));
        }
        // Calculate view point score
        std::vector<int> view_point_scores;
        for (int j = 0; j < view_space.size(); j++) {
          Point3D next_view_point = view_space[j];
          std::vector<Point3D> next_view_point_visible_points =
              view_space_points[j];
          // Calculate view point score
          int score = 0;
          for (Point3D point : next_view_point_visible_points) {
            std::tuple<int, int, int> voxelIndex =
                object_grid.pointToGridIndex(point);
            int x = std::get<0>(voxelIndex);
            int y = std::get<1>(voxelIndex);
            int z = std::get<2>(voxelIndex);
            // Check if the point is already reconstructed
            if (reconstructed_points_set.find(std::make_tuple(x, y, z)) ==
                reconstructed_points_set.end()) {
              score++;
            }
          }
          view_point_scores.push_back(score);
        }
        // Select the next best view (NBV) by the highest score of newly visible
        // points
        int max_score = 0;
        int max_score_index = 0;
        for (int j = 0; j < view_point_scores.size(); j++) {
          if (view_point_scores[j] > max_score) {
            max_score = view_point_scores[j];
            max_score_index = j;
          }
        }
        std::cout << "Next best view point for view point " << i
                  << " is view point " << max_score_index << " with score "
                  << max_score << std::endl;
        for (Point3D point : view_space_points[max_score_index]) {
          std::tuple<int, int, int> voxelIndex =
              object_grid.pointToGridIndex(point);
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          reconstructed_points_set.insert(std::make_tuple(x, y, z));
        }
        std::cout << "Reconstructed points number: "
                  << reconstructed_points_set.size() << std::endl;
        std::vector<Point3D> reconstructed_points;
        for (auto it = reconstructed_points_set.begin();
             it != reconstructed_points_set.end(); it++) {
          std::tuple<int, int, int> voxelIndex = *it;
          int x = std::get<0>(voxelIndex);
          int y = std::get<1>(voxelIndex);
          int z = std::get<2>(voxelIndex);
          Point3D voxel_center = object_grid.gridIndexToPoint(x, y, z);
          reconstructed_points.push_back(voxel_center);
        }
        std::string reconstructed_file =
            "../Output/" + object_name + "_res" + std::to_string(num_voxel) +
            "_vp" + std::to_string(i) + "_nbv" +
            std::to_string(max_score_index) + ".pcd";
        savePCD(reconstructed_file, reconstructed_points);

        // [mod] moved chmafer distance calculation CPU -> GPU
        // Calculate the chafer distance
        // double cd_before_nbv =
        //     chamfer_distance(all_visbile_points, view_point_visible_points);
        // double cd_after_nbv =
        //     chamfer_distance(all_visbile_points, reconstructed_points);
        double cd_before_nbv =
            chamfer_distance_gpu(all_visbile_points, view_point_visible_points);
        double cd_after_nbv =
            chamfer_distance_gpu(all_visbile_points, reconstructed_points);

        std::cout << "Number of visible points before NBV: "
                  << view_point_visible_points.size() << std::endl;
        std::cout << "Chamfer distance before NBV: " << cd_before_nbv
                  << std::endl;
        std::cout << "Number of visible points after NBV: "
                  << reconstructed_points.size() << std::endl;
        std::cout << "Chamfer distance after NBV: " << cd_after_nbv
                  << std::endl;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

  return 0;
}
