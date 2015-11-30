// from ELU.cu:

struct ELUupdateOutput_functor
{
  const float alpha_;

  ELUupdateOutput_functor(float alpha) : alpha_(alpha) {}

  /*__device__*/ void operator()(float* output, const float* input) const
  {
    *output = *input <= 0 ? (exp(*input)-1)*alpha_ : *input;
  }
};

struct ELUupdateGradInput_functor
{
  const float alpha_;

  ELUupdateGradInput_functor(float alpha) : alpha_(alpha) {}

  /*__device__*/ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

