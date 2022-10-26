# atari_rl
Implementation of Deep Reinforcement Learning Algorithms
    - Value Based: DQN with extensions(double, duel architecture, Prioretized Experience Replay)
    - Policy Based: PPO, PPO with exploration signal (RND, ICM)

You can run the project in Docker. To build the image:
```
sh build_image.sh
```

To start the container:
```
sh run_container.sh
```


Once, in the container shell:

```
python ./agents/deepq/run.py
```

or 

```
python ./agents/actorcritic/run.py
```
