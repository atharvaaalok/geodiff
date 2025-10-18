## 0.9.0
Add latent code functionality to represent geometry.
- Add latent code functionality to Pre-Aux nets.
- Add latent code functionality to all representations.
- Add test script for training with latent code using autodecoder framework.
    - Use NICE for the test script.
    - Fit two latent codes to fit two differently rotated squares.


## 0.8.0
Make Pre-Aux net modular by defining it separately from the invertible networks.
- Make Pre-Aux net modular for all representations.
- Change test scripts for all representations.
- Update documentation for all representations.


## 0.7.0
Add NeuralODE shape representation. This version update includes the following changes:
- NeuralODE parameterization.
    - Implement the NeuralODE class.
    - Add visualization method.
    - Add test script.
    - Add documentation.


## 0.6.0
Add NIGnet shape representation. This version update includes the following changes:
- NIGnet parameterization.
    - Add template architectures for monotonic networks MinMaxNet and SmoothMinMaxNet.
    - Implement the NIGnet class.
    - Add visualization method.
    - Add test script.
    - Add documentation.


## 0.5.0
Add RealNVP shape representation. This version update includes the following changes:
- RealNVP parameterization.
    - Implement the RealNVP class.
    - Add visualization method.
    - Add test script.
    - Add documentation.


## 0.4.0
Add NICE shape representation. This version update includes the following changes:
- NICE parameterization.
    - Add closed transforms for 2D and 3D.
    - Add template architectures for MLP and ResMLP.
    - Add Pre-Aux net class.
    - Implement the NICE class.
    - Add visualization method.
    - Add test script.
    - Add documentation.


## 0.3.0
Add CST shape representation. This version update includes the following changes:
- CST parameterization.
    - Implement the CST class.
    - Add visualization method.
    - Add test script.
    - Add documentation.
- Minor improvements to Hicks-Henne documentation.


## 0.2.0
Add Hicks-Henne shape representation. This version update includes the following changes:
- Install necessary dependencies
    - numpy, matplotlib and pytorch.
    - dev-dependencies - ipykernel.
- Implement loss functions.
    - Start with Chamfer loss.
- Hicks-Henne bump functions.
    - Implement the Hicks-Henne class.
    - Add visualization method.
    - Add test script.
    - Add documentation.


## 0.1.0
First version. Creates a basic project structure and sets up different workflows like Github Pages,
Github releases, PyPI publishing.