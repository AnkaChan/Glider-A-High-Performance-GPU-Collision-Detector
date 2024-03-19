#include <BVH/GPU_LBVH.cuh>
#include "PointCloudBVH.cuh"

#include <random>
#include <vector>
#include <thrust/random.h>
#include <algorithm>

#include "../Common/Timer.h"
#include "../Common/Types.h"
#include "../CuMatrix/CuMatrix/MatrixOps/CuMatrix.h"
#include "../CuMatrix/CuMatrix/MatrixOps/VectorTypes.h"

#define NUM_THEADS_TET_VOLUME 64

using CuMatrix::Vec3af;

void testVectorTypes();
void testBVH();

void pointCloudQuery() {
	//testVectorTypes();
	testBVH();
}

void checkVec3af(const Vec3af v, const GAIA::Vec3 g) {
    for (int i = 0; i < 3; i++) {
        assert(v[i] == g[i]);
    }
}

__global__ void computeTetVolume(const Vec3af* vertices, const int32_t* tets, float* volumes, size_t NT) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < NT) {
		const Vec3af a = vertices[tets[4 * idx]];
		const Vec3af b = vertices[tets[4 * idx + 1]];
		const Vec3af c = vertices[tets[4 * idx + 2]];
		const Vec3af d = vertices[tets[4 * idx + 3]];

		volumes[idx] = (1.0f / 6.0f) * ((b - a).dot((c - a).cross(d - a)));
	}
}

__global__ void computeTetVolume_unPadded(const float* vertices, const int32_t* tets, float* volumes, size_t NT)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NT) {
		const float* a = &vertices[3 * tets[4 * idx]];
		const float* b = &vertices[3 * tets[4 * idx + 1]];
		const float* c = &vertices[3 * tets[4 * idx + 2]];
		const float* d = &vertices[3 * tets[4 * idx + 3]];

        float b_a[3], c_a[3], d_a[3];

        CuMatrix::vec3Minus(b, a, b_a);
        CuMatrix::vec3Minus(c, a, c_a);
        CuMatrix::vec3Minus(d, a, d_a);

		float cross[3];
		CuMatrix::vec3CrossProduct(c_a, d_a, cross);

		volumes[idx] = (1.0f / 6.0f) * (CuMatrix::vec3DotProduct(b_a, cross));
	}

}

void testVectorTypes() {
    int numTests = 100000;
    for (size_t i = 0; i < numTests; i++)
    {
        GAIA::Vec3 a_ = GAIA::Vec3::Random();
        GAIA::Vec3 b_ = GAIA::Vec3::Random();

        Vec3af a(a_.data()), b(b_.data());

        checkVec3af(a + b, a_ + b_);
        checkVec3af(a - b, a_ - b_);
        checkVec3af(a.cross(b), a_.cross(b_));

        float rand = GAIA::Vec3::Random()[0];
        checkVec3af(a * rand, a_ * rand);
        checkVec3af(a / rand, a_ / rand);

        assert(abs(a.dot(b) - a_.dot(b_)) < 1e-5f);
        assert(abs(a.norm() - a_.norm()) < 1e-5f);

        a += b;
        checkVec3af(a, a_ + b_);
    }

    size_t NV = 10000000;
    thrust::host_vector<Vec3af> vertices(NV);
    thrust::host_vector<float> vertices_nonPad(3 * NV);

    for (size_t i = 0; i < NV; i++)
	{
		GAIA::Vec3 a_ = GAIA::Vec3::Random();
        vertices[i] = Vec3af(a_.data());
        vertices_nonPad[3 * i] = a_[0];
        vertices_nonPad[3 * i + 1] = a_[1];
        vertices_nonPad[3 * i + 2] = a_[2];
	}

    thrust::device_vector<Vec3af> vertices_d = vertices;
    thrust::device_vector<float> vertices_nonPad_d = vertices_nonPad;

    size_t NT = 10000000;
    thrust::host_vector<int32_t> tets(NT*4);

    std::mt19937 mt(123456789);

    int32_t vertexRange = 64;

    for (size_t i = 0; i < NT; i++)
    {

        float r = i / (float)NT;

        int32_t rangeLow = (int32_t)(r * NV) - vertexRange;
        rangeLow = std::max(rangeLow, 0);
        int32_t rangeHigh = (int32_t)(r * NV) + vertexRange;
        rangeHigh = std::min(rangeHigh, (int32_t)NV - 1);

        std::uniform_int_distribution<int32_t> uni(rangeLow, rangeHigh);

        tets[4 * i] = uni(mt);
        tets[4 * i + 1] = uni(mt);
        tets[4 * i + 2] = uni(mt);
        tets[4 * i + 3] = uni(mt);
    }
    thrust::device_vector<int32_t> tets_d = tets;

    printf("Size of Vec3af: %zu\n", sizeof(Vec3af));

    printf("Address of vertices_d: %p | vertices: %p\n", vertices_d.data().get(), vertices.data());
    printf("Address of vertices_nonPad_d: %p | vertices_nonPad: %p\n", vertices_nonPad_d.data().get(), vertices_nonPad_d.data());

    thrust::host_vector<float> volume(NT);
    thrust::host_vector<float> volume_nonPad(NT);

    thrust::device_vector<float> volume_d = volume;
    thrust::device_vector<float> volume_nonPad_d = volume_nonPad;

    size_t numSamples = 100;
    // warm-ups
    for (size_t i = 0; i < numSamples; i++)
    {
        computeTetVolume KERNEL_ARGS2((NT + NUM_THEADS_TET_VOLUME - 1) / NUM_THEADS_TET_VOLUME, NUM_THEADS_TET_VOLUME) 
            (vertices_d.data().get(), tets_d.data().get(), volume_d.data().get(), NT);
    }

    for (size_t i = 0; i < numSamples; i++)
    {
        computeTetVolume_unPadded KERNEL_ARGS2((NT + NUM_THEADS_TET_VOLUME - 1) / NUM_THEADS_TET_VOLUME, NUM_THEADS_TET_VOLUME) 
            (vertices_nonPad_d.data().get(), tets_d.data().get(), volume_nonPad_d.data().get(), NT);
    }
    cudaDeviceSynchronize();

    double computeTetVolumeTime = 0;
    TICK(computeTetVolumeTime);
    for (size_t i = 0; i < numSamples; i++)
    {
        computeTetVolume KERNEL_ARGS2((NT + NUM_THEADS_TET_VOLUME - 1) / NUM_THEADS_TET_VOLUME, NUM_THEADS_TET_VOLUME)
            (vertices_d.data().get(), tets_d.data().get(), volume_d.data().get(), NT);
    }
    cudaDeviceSynchronize();
    TOCK(computeTetVolumeTime);

    double computeTetVolumeTime_unPadded = 0;
    TICK(computeTetVolumeTime_unPadded);
    for (size_t i = 0; i < numSamples; i++)
    {
        computeTetVolume_unPadded KERNEL_ARGS2((NT + NUM_THEADS_TET_VOLUME - 1) / NUM_THEADS_TET_VOLUME, NUM_THEADS_TET_VOLUME) 
            (vertices_nonPad_d.data().get(), tets_d.data().get(), volume_nonPad_d.data().get(), NT);
    }
    cudaDeviceSynchronize();
    TOCK(computeTetVolumeTime_unPadded);

    volume = volume_d;
    volume_nonPad = volume_nonPad_d;

	for (size_t i = 0; i < NT; i++)
	{
		assert(abs(volume[i] - volume_nonPad[i]) < 1e-5f);
	}

    printf("computeTetVolumeTime: %f\n", computeTetVolumeTime);
    printf("computeTetVolumeTime_unPadded: %f\n", computeTetVolumeTime_unPadded);

    std::cout << "All tests passed!" << std::endl;
}


void testBVH() {

    // constexpr std::size_t N = 100000;
    constexpr std::size_t N = 10;
    std::vector<float> ps1(N * 3 );
    std::vector<float> ps2(N * 3 );

    std::mt19937 mt(123456789);
    std::uniform_real_distribution<float> uni(0.0, 1.0);

    for (auto& p : ps1)
    {
        p = uni(mt);
    }

    for (auto& p : ps2)
    {
        p = uni(mt);
    }

    lbvh::PointCloudBVH<float> pointCloudBVH;

    std::vector<float*> pointClouds = { ps1.data(), ps2.data()};
    std::vector<size_t> pointCloudSizes = { N, N };

    pointCloudBVH.initialize(pointClouds, pointCloudSizes);

    //const auto bvh_dev = pointCloudBVH.get_device_repr();

    std::cout << "testing query_device: intersections ...\n";

    thrust::host_vector<float> queryPositions(ps1.begin(), ps1.end());

    pointCloudBVH.setQueryPositions(queryPositions);
    pointCloudBVH.queryOverlaps();



    //std::cout << "testing query_device: overlap ...\n";
    //thrust::for_each(thrust::device,
    //    thrust::make_counting_iterator<std::size_t>(0),
    //    thrust::make_counting_iterator<std::size_t>(N),
    //    [bvh_dev] __device__(std::size_t idx) {
    //    unsigned int buffer[10];
    //    const float4& self = bvh_dev.aabbs[idx + N + idx -1].lower;
    //    const float  dr = 0.1f;
    //    for (unsigned int j = 0; j < 10; ++j)
    //    {
    //        buffer[j] = 0xFFFFFFFF;
    //    }
    //    //const float r = dr * i;
    //    const float r = dr / N;
    //    lbvh::aabb<float> query_box;
    //    query_box.lower = make_float4(self.x - r, self.y - r, self.z - r, 0);
    //    query_box.upper = make_float4(self.x + r, self.y + r, self.z + r, 0);
    //    const auto num_found = lbvh::query_device(
    //        bvh_dev, lbvh::overlaps(query_box), buffer, 10);

    //    for (unsigned int j = 0; j < 10; ++j)
    //    {
    //        const auto jdx = buffer[j];
    //        if (j >= num_found)
    //        {
    //            assert(jdx == 0xFFFFFFFF);
    //            continue;
    //        }
    //        else
    //        {
    //            assert(jdx != 0xFFFFFFFF);
    //            assert(jdx < bvh_dev.num_objects);
    //        }
    //        const auto other = bvh_dev.objects[jdx];
    //        assert(fabsf(self.x - other.x) < r); // check coordinates
    //        assert(fabsf(self.y - other.y) < r); // are in the range
    //        assert(fabsf(self.z - other.z) < r); // of query box
    //    }
    //    return;
    //});

}