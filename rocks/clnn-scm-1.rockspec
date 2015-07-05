package = "clnn"
version = "scm-1"

source = {
   url = "git://github.com/hughperkins/clnn.git",
}

description = {
   summary = "Torch OpenCL Neural Network Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/hughperkins/clnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "cltorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
]],
   install_command = "cd build"
}
