from safetensors import safe_open


def main() -> None:
    model_path = "model/model.safetensors"
    with safe_open(model_path, framework="np") as f:
        keys = f.keys()
        print(f"Loaded: {model_path}")
        print(f"Tensor count: {len(keys)}")
        if keys:
            print(f"First key: {keys[0]}")


if __name__ == "__main__":
    main()
