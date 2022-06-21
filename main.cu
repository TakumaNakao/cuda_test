#include <stdio.h>
#include <cuda.h>

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class TestClass {
public:
    TestClass()
    {

    }
    __device__ void print()
    {
        printf("%d\n", n_);
    }
    __host__ void set(int n)
    {
        n_ = n;
    }
private:
    int n_;
};

__global__ void hello(TestClass a, TestClass b, TestClass c) 
{
    a.print();
    b.print();
    c.print();
}

int main() {
    TestClass a;
    TestClass b;
    TestClass c;
    a.set(1);
    b.set(2);
    c.set(3);
    hello << < 2,2 >> > (a,b,c);
    cudaDeviceSynchronize();
    return 0;
}