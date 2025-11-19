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

        RMSProp(float lr_, float alpha_ = 0.9, float eps_ = 0.000000001)
            : lr(lr_), alpha(alpha_), eps(eps_) {}
    };
    
    using OptimVariant = std::variant<
    SGD,
    Momentum,
    RMSProp
    >;
}
