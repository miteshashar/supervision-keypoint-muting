# Roboflow Supervision - Keypoint Muting Experiment

This is an attempt to understand how Roboflow's [Supervision](https://github.com/roboflow/supervision) works.

The experiment was inspired from the issue opened on [roboflow/supervision#1676](https://github.com/roboflow/supervision/issues/1676).

The utility of the proposed method `with_threshold` in `KeyPoints` was not clear to me, as in how setting low-threshold keypoints to `0` helps in vizualizing skeletons with low confidence.

Hence, I tried to understand it by illustrating it.

## Output

### Example 1

**[Original Image](https://pixabay.com/photos/soccer-competition-football-stadium-3311817/)**
!(examples/1.jpg "Example 1 – Original Image)

**Output**
!(examples/1.webm "Example 1 – Output)

### Example 2

**[Original Image](https://pixabay.com/photos/ski-skier-sports-downhill-slope-79564/)**
!(examples/2.jpg "Example 2 – Original Image)

**Output**
!(examples/2.webm "Example 2 – Output)

### Example 3

**[Original Image](https://pixabay.com/photos/runners-male-sport-run-athlete-373099/)**
!(examples/3.jpg "Example 3 – Original Image)

**Output**
!(examples/3.webm "Example 3 – Output)

### Example 4

**[Original Image](https://pixabay.com/photos/sumo-wrestler-athlete-wrestler-hall-3196753/)**
!(examples/4.jpg "Example 4 – Original Image)

**Output**
!(examples/4.webm "Example 4 – Output)

Images used are free and permitted. Credits: [Pixabay](https://pixabay.com)