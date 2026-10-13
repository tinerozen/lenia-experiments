# Lenia experiments

Experiments with [Lenia](https://en.wikipedia.org/wiki/Lenia)-style continuous cellular automata, plus a few related simulations. The code and comments are in Slovenian.

Every script is standalone. It runs the simulation and plays it as a matplotlib animation. Most of them then save an image to `test.png`, and the Lenia ones also plot total mass. Run them from the repo root:

```bash
pip install -r requirements.txt
python lenia.py
```

## Structure

```
lenia.py                      first Lenia version; set VARIANT at the top:
                                "basic", "constant_mass" or "continuous"
actual_lenia.py               reworked version (N=128, 23×23 ring kernel, random drift in the growth center)
actual_lenia_flow.py          moves mass toward the highest-affinity neighbour instead of adding growth
actual_lenia_temperature.py   adds a "temperature" (global mass) that sharpens/widens growth, plus a flashing bar

optimization/                 attempts to speed up actual_lenia_temperature.py
  opt1_numba_joblib.py
  opt2_numba_joblib_clip.py
  opt3_numba_threads.py
  opt4_numba_class.py         saves images/optimized_result.png
  actual_lenia_temperature.ipynb

other/
  conway.py                   Game of Life via convolution
  organic_life_sim.py         precursor to lenia.py: survive/revive rule on two box kernels
  fluid.py                    diffusion with a hand-written Gaussian blur
  fluid_convolve.py           same idea with scipy.ndimage.convolve
  gravity.py                  unfinished: disipacija() has no body, so it doesn't run
  gradient_test.py            checks how np.gradient behaves

misc/                         unrelated side tests
  snake.py                    pygame snake
  julia_vs_python/            factorial timing in Julia vs Jupyter

images/                       example outputs
```
