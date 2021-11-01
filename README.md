# Webots Web Log Interface
A python library used to interact with webots robocup game web logs

## Installation

```bash
pip3 install webots-web-log-interface
```

## Documentation

You can find the interface documentation [here](https://bit-bots.github.io/webots-web-log-interface/html/webots_web_log_interface/interface.html).

## Examples

Download example data

```bash
mkdir data
cd data
wget https://games.bit-bots.de/k-ko-sf2/K-KO-SF2.json
wget https://games.bit-bots.de/k-ko-sf2/K-KO-SF2.x3d
cd ..
```

Now you are able to use the interface

```python
from webots_web_log_interface.interface import WebotsGameLogParser

gp = WebotsGameLogParser(log_folder="data")

# Now some examples
# Get ball
ball = gp.x3d.get_ball_id()
# Get velocities for ball
print(gp.game_data.get_velocity_vectors_for_id(ball))
# Get player names
print(gp.x3d.get_player_names())
# Plot player paths
gp.plot_player_paths()
```

## Build it yourself

```bash
git clone https://github.com/bit-bots/webots-web-log-interface.git
cd webots-web-log-interface

poetry install
poetry shell
```
