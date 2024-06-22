import fiftyone as fo
import fiftyone.zoo as foz


if __name__ == "__main__":
    # https://docs.voxel51.com/user_guide/app.html#creating-a-session
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.launch_app(dataset, remote=True)
    session.wait()
