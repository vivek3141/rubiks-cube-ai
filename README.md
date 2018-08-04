# Rubik's Cube AI
The Rubik's Cube is the most popular toy right now. This
program uses deep-q learning and OpenAI baselines to learn how 
to solve the cube.

## Requirements
You can run `sudo sh requirements.sh` to get all the requirements
<br />
If on windows, run `requirements`

## How it works
This program uses Deep Q learning and OpenAI's gym and baselines. I have experimented
with Keras also, however I found that baselines gave better results.

## Usage
To train, run 
<br />
`python3 main.py train`
<br /><br />
To run, run <br />
`python3 main.py run`

## Debugging
Open main.py in your editor and you can edit all the variables for the classes `Run` and `Train`.

## Credits
@Robin Chiu for writing the base of the env. I fixed a few issues which is why it has been provided.