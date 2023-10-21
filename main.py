paths = {
    "positive": [
        "./data/Test_1BEatHeadStill.csv",
        "./data/Test_1CChewGuava.csv",
        "./data/Test_2AEatMoving.csv",
        "./data/Test_2CEatNhan.csv",
    ],
    "negative": [
        "./data/Test_1ANoeatHeadStill.csv",
        "./data/Test_2ANoeatMoving.csv",
    ]
}

from dataset import Dataset
dataset = Dataset(paths, window_size=80, batch_size=32, shuffle=True).dataset

from model import create_model
model = create_model(dataset_shape=dataset.element_spec[0].shape, verbose=5)
# TODO: wth did it run so many times?

from model import fit  
fit(model, dataset, epochs=50)