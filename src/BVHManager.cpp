#include <vector>

#include "BVHManager.h"
#include "helper_math.h"
#include "RayGenerator.h"


#include "ValidationKernels.h"
#include "EmbreeBVHBuilder.h"
#include "GPUBVHConverter.h"

#include "TraversalKernelBVH2.h"
#include "TraversalKernelCWBVH.h"

void BVHManager::buildBVH2(RayGenerator& rg)
{
  printf("Building BVH2\n");

  #if 0 // BVH-ESC (Early Split Clipping)
    BVH2Node* RootBVHNode = BuildBVH2AABB((int)IndexBuffer.size(), [&](int PrimitiveIndex, float3& OutLower, float3& OutUpper)
    {
      OutLower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
      OutLower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
      OutLower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
      OutUpper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
      OutUpper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
      OutUpper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
    });
  #else // SBVH
    BVH2Node* RootBVHNode = BuildBVH2Triangle(rg.IndexBuffer, rg.VertexBuffer);
  #endif

    GPUBVHIntermediates BVHData;

    printf("Converting to GPU BVH2\n");

    ConvertToGPUBVH2(RootBVHNode,
      [&](int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)
    {
      float4 V0, V1, V2;
      WoopifyTriangle(
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].x],
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].y],
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].z],
        V0, V1, V2
      );
      InlinedPrimitives.push_back(V0);
      InlinedPrimitives.push_back(V1);
      InlinedPrimitives.push_back(V2);
    },
      BVHData);

    RootBVHNode->Release();

    float4* cudaBVHNodeData;
    float4* cudaInlinedPrimitives;
    int* cudaPrimitiveIndices;
    cudaMalloc(&cudaBVHNodeData, BVHData.BVHNodeData.size() * sizeof(float4));
    cudaMalloc(&cudaInlinedPrimitives, BVHData.InlinedPrimitives.size() * sizeof(float4));
    cudaMalloc(&cudaPrimitiveIndices, BVHData.PrimitiveIndices.size() * sizeof(int));

    cudaMemcpy(cudaBVHNodeData, BVHData.BVHNodeData.data(), BVHData.BVHNodeData.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaInlinedPrimitives, BVHData.InlinedPrimitives.data(), BVHData.InlinedPrimitives.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPrimitiveIndices, BVHData.PrimitiveIndices.data(), BVHData.PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    rtBindBVH2Data(
      cudaBVHNodeData,
      cudaInlinedPrimitives,
      cudaPrimitiveIndices
    );
  }

  void BVHManager::buildCWBVH(RayGenerator& rg)
  {
    printf("Building CWBVH\n");

  #if 0 // BVH-ESC (Early Split Clipping)
    BVH8Node* RootBVHNode = BuildBVH8AABB((int)IndexBuffer.size(), [&](int PrimitiveIndex, float3& OutLower, float3& OutUpper)
    {
      OutLower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
      OutLower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
      OutLower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
      OutUpper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
      OutUpper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
      OutUpper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
    });
  #else // SBVH
    BVH8Node* RootBVHNode = BuildBVH8Triangle(rg.IndexBuffer, rg.VertexBuffer);
  #endif

    GPUBVHIntermediates BVHData;

    printf("Converting to GPU CWBVH\n");

    ConvertToGPUCompressedWideBVH(RootBVHNode,
      [&](int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)
    {
      float4 V0, V1, V2;
      WoopifyTriangle(
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].x],
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].y],
        rg.VertexBuffer[rg.IndexBuffer[PrimitiveIndex].z],
        V0, V1, V2
      );
      InlinedPrimitives.push_back(V0);
      InlinedPrimitives.push_back(V1);
      InlinedPrimitives.push_back(V2);
    },
      BVHData);

    RootBVHNode->Release();

    float4* cudaBVHNodeData;
    float4* cudaInlinedPrimitives;
    int* cudaPrimitiveIndices;
    cudaMalloc(&cudaBVHNodeData, BVHData.BVHNodeData.size() * sizeof(float4));
    cudaMalloc(&cudaInlinedPrimitives, BVHData.InlinedPrimitives.size() * sizeof(float4));
    cudaMalloc(&cudaPrimitiveIndices, BVHData.PrimitiveIndices.size() * sizeof(int));

    cudaMemcpy(cudaBVHNodeData, BVHData.BVHNodeData.data(), BVHData.BVHNodeData.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaInlinedPrimitives, BVHData.InlinedPrimitives.data(), BVHData.InlinedPrimitives.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPrimitiveIndices, BVHData.PrimitiveIndices.data(), BVHData.PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    rtBindCWBVHData(
      cudaBVHNodeData,
      cudaInlinedPrimitives,
      cudaPrimitiveIndices
    );
  }
