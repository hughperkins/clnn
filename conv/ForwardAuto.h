class ForwardAuto : public Forward {
public:
    int num;
    int *milliseconds;
    bool *valid;
    int chosenIndex;
    Forward **instances;
    int nextIndex;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ForwardAuto(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~ForwardAuto();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper,
    CLWrapper *biasWrapper, CLWrapper *outputWrapper);

    // [[[end]]]
};

