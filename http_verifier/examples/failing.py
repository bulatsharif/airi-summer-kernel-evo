print("stdout before failure", flush=True)
raise RuntimeError("intentional remote smoke-test failure")
