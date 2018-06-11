# Deep Q-Network with Experience Replay
To train a model, run
```python
python3 main.py --env [environment_name] --model [model_name] --log [log_directory]
```
The available models are `linear`, `dqn`, `dueling` and `dueling_conv`.
`dueling_conv` is used for `SpaceInvaders-v0` environment only.
Use `--replay` to turn on experience replay.

To test a model, run
```python
python3 main.py --test --env [environment_name] --checkpoint [saved_checkpoint_path]
```

For other options and hyperparameters, run
```python
python3 main.py --help
```
