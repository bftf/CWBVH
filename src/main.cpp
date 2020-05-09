#include "RayGenerator.h"
#include "BVHManager.h"
#include "RayTraceManager.h"

#include <iostream>
#include <string>
#include <assert.h>

int main()
{
  const std::string model_base_path = "/home/francois/Documents/UBC/masters/CWBVH/"; // pc-12
  // const std::string model_base_path = "/home/francois/Documents/RayTracing/models/"; // pc-16

  const std::string ply_base_path = "./ply_files/";
  const std::string ray_base_path = "./ray_files/"; 

  const std::string model_name = "teapot";
  const std::string obj_path_addition = "teapot.obj";

  // const std::string model_name = "sponza";
  // const std::string obj_path_addition = "Sponza/models/sponza.obj";

  // const std::string model_name = "dragon";
  // const std::string obj_path_addition = "Dragon/dragon.obj";

  // const std::string model_name = "san-miguel";
  // const std::string obj_path_addition = "San_Miguel/san-miguel.obj";

  const std::string out_ply_path = ply_base_path + model_name;
  const std::string model_path = model_base_path + obj_path_addition;

  /* Set Up the Ray Generator */
  RayGenerator rg = RayGenerator(4, 4, 0.1, 10); /*spp, spt, t_min, t_max*/

  /* Load the model */ 
  int triangle_count = rg.loadModelOBJ(model_path);
  assert(triangle_count > 0);
  printf("Done loading obj. Tri count: %i \n", triangle_count);

  /*Build the acceleration structures*/
  BVHManager bvh_manager = BVHManager();

  bvh_manager.buildBVH2(rg);
  bvh_manager.buildCWBVH(rg);


  /* Generate rays - uses points and normals -> this takes a while */
  int spp_count;
  spp_count = rg.generatePointsAndNormals_spt(1000000);
  spp_count = rg.generate_detangled_spp();
  // int sample_count = rg.generate_detangled_spp();
  printf("Done generating rays detangled. SPP count: %i \n", spp_count);
  
  /* Apply ray sorting + save rays */
  rg.raySorting(rg.random_shuffle);
  rg.saveRaysToFile(ray_base_path, model_name + "_detangled_random");

  /* Upload rays to the GPU*/
  rg.uploadRaysToGPU();

  /* Trace the rays */ 
  RayTraceManager rt_manager = RayTraceManager(rg);
  rt_manager.traceCWBVH(rg);
  rt_manager.traceAila(rg);

  printf("Done, traced %i rays \n", rg.getRayCount());

  exit(0);

  /* Shuffle and save to file -> this should only be done once
  rg.raySorting(rg.random_shuffle);
  rg.saveRaysToFile(ray_base_path, model_name + "_randomshuffle");

  rg.raySorting(rg.origin);
  rg.saveRaysToFile(ray_base_path, model_name + "_origin");

  rg.raySorting(rg.random_shuffle);
  rg.raySorting(rg.direction);
  rg.saveRaysToFile(ray_base_path, model_name + "_direction");
  
  rg.readRaysFromFile(ray_base_path + "sponza_randomshuffle_16785088_12_10.ray_file", 16785088);
  rg.raySorting(rg.origin_chunk);
  rg.saveRaysToFile(ray_base_path, model_name + "_origin_chunk_8192");

  printf("Done sorting rays\n");
  exit(0);
  */

  // rg.readRaysFromFile(ray_base_path + "sponza_randomshuffle_16785088_12_10.ray_file", 16785088);
  // rg.readRaysFromFile(ray_base_path + "sponza_direction_16785088_12_10.ray_file", 16785088);
  // rg.readRaysFromFile(ray_base_path + "sponza_origin_16785088_12_10.ray_file", 16785088);

  // rg.readRaysFromFile(ray_base_path + "teapot_origin_1spp_64spt_origin_404480_18_10.ray_file", 404480);
  // rg.readRaysFromFile(ray_base_path + "sponza_origin_chunk_8192_16785088_13_10.ray_file", 16785088);
  // rg.readRaysFromFile(ray_base_path + "sponza_direction_16785088_12_10.ray_file", 200000);
  // rg.readRaysFromFile(ray_base_path + "sponza_origin_16785088_12_10.ray_file", 200000);

  // rg.readRaysFromFile(ray_base_path + "dragon_500000_30_9.ray_file", 200000);
  // rg.readRaysFromFile(ray_base_path + "teapot_100_28_9.ray_file", 100);
  
  // rg.fillWithListOfRays(); // fill custom list of rays for debugging
  
  rg.uploadRaysToGPU();




  /*
  for (int i=0; i < 5000; i++)
  {
    rg.uploadSingleRayToGPU(i);
    rt_manager.traceCWBVHSingleRay(rg);
    printf("Done tracing ray number: %i\n", i);
  }
  */

  /* OptiX */
  // rt_manager.traceOptiXPrime(rg);

  /* Aila */    
  // rt_manager.traceAila(rg); 

  /* CWBVH */
  rt_manager.traceCWBVH(rg);
  
  rt_manager.evaluateAndPrintForPLYVisualization(rg, out_ply_path); // prints outcome from kernels for visualization (make sure 8, 8 is enabled when reading from file!)
  // rt_manager.debugging(rg); // for debugging - prints output of kernels into command line
  
  printf("Done, traced %i rays \n", rg.getRayCount());
  return 0;
}

// aamodt-pc12
// /home/francois/cuda_10_0/lib64:/home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/
// /home/francois/Documents/gpgpu-sim_distribution/lib/gcc-5.4.0/cuda-10000/debug:/home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/
// /home/francois/Documents/gpgpu-sim_distribution/lib/gcc-5.4.0/cuda-10000/release:/home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/

// /home/francois/Documents/RayTracing/embree/include:/home/francois/cuda_10_0/lib64:/home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/

// aamodt-pc16
// /home/common/modules/cuda/10.0.130/lib64:/home/francois/Documents/embree/build/
// /home/francois/Documents/gpgpu-sim_cuda10/lib/gcc-7.4.0/cuda-10000/debug:/home/francois/Documents/embree/build/
// /home/francois/Documents/gpgpu-sim_cuda10/lib/gcc-7.4.0/cuda-10000/release:/home/francois/Documents/embree/build/
