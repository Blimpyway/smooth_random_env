import multiprocessing as mp
from functools import partial

# 2. The Worker Loop (Runs in child process)
def env_worker(pipe, env_fn):
    env = env_fn()  # The env is instantiated here
    try:
        while True:
            command, data = pipe.recv()
            if command == "step":
                result = env.step(data)
                pipe.send(result)
            elif command == "reset":
                result = env.reset()
                pipe.send(result)
            elif command == "close":
                env.close()
                break
    except EOFError:
        pass
    finally:
        pipe.close()

# 3. The Proxy Handle (Used in main process)
class EnvProxy:
    def __init__(self, env_fn):
        self.parent_pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=env_worker, args=(child_pipe, env_fn))
        self.process.start()

    def step(self, action):
        self.parent_pipe.send(("step", action))
        return self.parent_pipe.recv()

    def reset(self):
        self.parent_pipe.send(("reset", None))
        return self.parent_pipe.recv()

    def close(self):
        self.parent_pipe.send(("close", None))
        self.process.join()

# 4. Usage Example
if __name__ == "__main__":
    import time
    # 1. Your Custom Environment
    class MyCustomEnv:
        def __init__(self, ename):
            self.name  = ename
            self.steps_taken = 0

        def my_state(self):
            return f"{self.name} steps {self.steps_taken}"

        def reset(self):
            self.steps_taken = 0
            return self.my_state()

        def step(self, action):
            time.sleep(1) # pretend doing hard work
            self.steps_taken += 1
            # Simple dummy logic
            reward = 1.0
            terminated = self.steps_taken >= 5
            truncated = False
            info = {}
            return self.my_state(), reward, terminated, truncated, info

        def close(self):
            print(f">>> Environment {self.name} shutting down...")

    # Create the factory function with your arguments
    make_env1 = partial(MyCustomEnv, ename = "Mary env")
    make_env2 = partial(MyCustomEnv, ename = "Jane env")

    # Initialize two independent proxies (running on different cores)
    env1 = EnvProxy(make_env1)
    env2 = EnvProxy(make_env2)

    # You can step them independently!
    print(f"Env 1 Reset: {env1.reset()}")
    print(f"Env 2 Reset: {env2.reset()}")

    while True:
        done = 0
        for env in [env1,env2]:
            obs, reward, terminated, truncated, info = env.step("dummy action") 
            print(obs)
            done += terminated
        if done == 2: 
            break

    print("Will close in 2 seconds..." )
    time.sleep(2)
    # Clean up
    env1.close()
    env2.close()


