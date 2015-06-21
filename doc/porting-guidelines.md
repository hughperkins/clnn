# Porting guidelines

To port a new layer from cunn, proceed approximately as follows:

## Add the files, and get them building

* make sure that `cunn` is checked out to directory `cunn`, at the same level as `clnn` directory
* change into `clnn` directory
* create a directory `port`
* run `python util/port.py`, which will do a first-cut port of the cuda files from `../cunn` directory into the `port` subdirectory
* use meld or similar to copy the two or so files from the desired layer across into the `clnn` directory
* add the .cpp file to CMakeLists.txt
* change the init function, at hte bottom of the layer's .cpp file to not be static
* add a call to the layer's init function to the `init.cpp` file
* add some includes to the top of the .cpp file:
```
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"

#include <iostream>
#include <string>
using namespace std;
```
* comment out any cuda-stuff in the .cpp file, and add `THError("Not implemented");` in their place
* try building, and fix any build errors

## Port the .cl file

In the .cl file:
* replace each `float * foo` kernel parameter with `global float *foo_data, int foo_offset`
* at the start of the kernel, for each of these parameters, put:
```
global float *foo = foo_data + foo_offset;
```
* put `global` in front of any float * variables that are derived from these variables
* put `local` in front of any float * variables derived from any float variables

# Stringify the kernel into the .cpp file

* Add a `stringify` section to the bottom of the .cpp file. It should look something like:
```
std::string MyNewLayer_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "MyNewLayer.cl" )
  // ]]]
  // [[[end]]]
  return kernelSource;
}
```
* change the bit saying MyNewLayer.cl to have the actual name of the .cl file
* change the name of the method to replace `MyNewLayer` with the actual name of the layer
* cd into `build` directory, run `ccmake ..`, and change option `DEV_RUN_COG` to `ON`, and do `configure` and `generate`
* rebuild => the bottom of the .cpp file should now contain the .cl source code, as a c++ std::string
* copy the declaration of this method to the top of the .cpp file

# Call the kernel

Calling the kernel comprises the following parts:
* create a kernel templater, something like
```
TemplatedKernel kernelBuilder(THClState_getCl(state));
```
* If there are any templated parameter to replace (not discussed in this doc yet), you'll need to pass those to the templater now
* create the kernel
  * you need to create a unique name.  This will be used to lookup the compiled kernel, and re-use.  If it is not sufficiently unique, it will collide with other kernels of the same name, and the wrong kernel will be called ;-)
  * give the name of the cl file (this wont affect anything, just used for error messages; not so critical)
  * you need to provide the name of the stringify function you created above
  * you need to provide the exact name of the kernel function; if it's wrong, then the kernel wont be able to be run
```
    std::string uniqueName = __FILE__ "maxpool";
    CLKernel *kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      SpatialMaxPooling_getKernelTemplate(), "maxpool");
```
* create a THClKernels object, from the kernel object you created just now
```
    THClKernels k(state, kernel);
```
* pass in parameters
```
    k.in(input);
    k.out(output);
    k.out(indices);
    k.in((int)(nbatch*nInputPlane*nOutputCols*nOutputRows));
    k.in((int)0);
    k.in((int)nInputPlane);
    k.in((int)nInputRows);
    k.in((int)nInputCols);
    k.in((int)kH);
    k.in((int)kW);
    k.in((int)dH);
    k.in((int)dW);
```
* call the kernel :-)
```
    k.run(blocks, threads);
```

