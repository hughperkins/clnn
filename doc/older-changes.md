# Older changes

* 10th August:
  * Improve error message when out of memory, ie will say it ran out of memory, rather than say 'c++ exception' now, in many common cases
  * SpatialMaxPooling can now handle pooling size and stride are different, as long as half the pooling size is no more than stride
  * Added SpatialAveragePooling for case where input size equals filter size, or filter size equals stride size
* 22nd July:
  * Performance improvements in underlying [cltorch](https://github.com/hughperkins/cltorch) mean that times for [char-rnn](http://github.com/karpathy/char-rnn) are now around 2-3 times faster on NVIDIA and AMD GPUs
* 6th July:
  * lots of new activations added: `Sqrt`, `Square`, `Exp`, `Abs`, `LogSigmoid`, `HardTanh`  (provided by Sergey Zagoruyko)
  * SpatialMaxPooling:
    * added implicit floor max pooling (provided by Sergey)
    * added 3d forward (from Sergey)
  * added tests from cunn (thank you Sergey)
  * bug fixes:
    * SpatialConvolutionMM updated to match current nn (Sergey)
    * fixed bug in ReLU for in-place forward
* 27th June:
  * mild perf improvement to LogSoftMax layer
  * removed FullyConnected for now
  * mild perf improvement to Narrow layer
  * huge perf improvement :-)  Please update to latest version of [cltorch](http://github.com/hughperkins/cltorch) (should be at least commit 2f1e3e758fb or later)
* 26th June:
  * fixed bug in Sigmoid, which wasnt resizing correctly
* 25th June:
  * added tests for CMulTable and CAddTable, which pass
  * added test for Narrow, which passes
  * fix bug in cmakelists.txt, which meant that installation didnt work (it ran ok for me, so I didnt notice...)
  * Dropout working now
* 24th June:
  * Added ClassNLLCriterion layer (and unit tests for this)
* 23rd June:
  * Added LogSoftMax layer (and unit test for this)
* 22nd June:
  * Checked that SpatialConvolutionMM gives same results using clnn, compared with cunn
  * Checked that SpatialMaxPooling gives same results using clnn, compared with nn
  * Added ReLU, which was already marked as added but ... wasnt :-P  but now is :-) )
* 21st June:
  * Got SpatialConvolutionMM and SpatialMaxPooling running
  * Ran Soumith benchmarks on SpatialConvolutionMM, for clnn and cunn, on NVidia 940M

