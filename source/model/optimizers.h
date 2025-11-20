#pragma once

#include <concepts>
#include <variant>

namespace wolf {

    struct SGD { // Stochastic Gradient Descent (Default)
        float lr;

        explicit SGD(float lr_) // Learn Rate
         : lr(lr_) {} 
    };

    struct Momentum {
        float lr;
        float mu;

        Momentum(float lr_, float momentum = 0.5) // Learn rate and momentum
            : lr(lr_), mu(momentum) {}
    };

    struct RMSProp {
        float lr;
        float alpha;
        float eps;

        RMSProp(float lr_, float alpha_ = 0.9f, float eps_ = 1e-8f)
            : lr(lr_), alpha(alpha_), eps(eps_) {}
    };

    struct Adam { // Adaptive Moments
        float lr;
        float beta1;
        float beta2;
        float eps;

        Adam(float lr_, float beta1_ = 0.9f, float beta2_ = 0.99f, float eps_ = 1e-8f) 
            : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_) {}
    };
    
    using OptimVariant = std::variant<
    SGD,
    Momentum,
    RMSProp,
    Adam
    >;
}
