"""Scene-admission state machine, independent of the policy loop.

No inference, env.step, Local begin, simulator import or production hooks. The
recover callback must explicitly return a safe environment; None is unsafe.
The native adapter must certify resource safety, not just catch Exception.
"""

import uuid


def boolean(value):
    if type(value) is bool or (type(value).__module__ == "numpy" and type(value).__name__ in ("bool", "bool_")):
        return bool(value)
    raise ValueError("initial success predicate must return boolean")


def admit_scene(store, attempt, env, recover):
    while True:
        state = store._admission_state(attempt["task"], attempt["episode"])
        if state["status"] in ("skipped_init_error", "scene_admission_exhausted", "invalid"):
            return state, env
        store.begin_initialization(attempt)
        stage = "reset"
        try:
            env.reset()
            stage = "initial_success"
            initial = boolean(env._check_success())
        except Exception as error:
            try:
                restored = recover(error, stage, env)
            except Exception:
                restored = None
            state = store.initialization_outcome(attempt, error=error, stage=stage, safe_to_retry=restored is not None)
            if restored is not None:
                if restored is not env:
                    store._event(
                        attempt,
                        "ENV_REBUILT",
                        {"environment_id": uuid.uuid4().hex, "reason": "explicit safe recovery rebuilt environment"},
                    )
                env = restored
        else:
            state = store.initialization_outcome(attempt, initial_success=initial)
        if state["status"] in ("admitted", "skipped_init_error", "scene_admission_exhausted", "invalid"):
            return state, env
