# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the improved and cleaned up library script to create models and reports for **Predict Customer Churn**.
The code was in the `churn_notebook.ipynb` as a source for the project itself.

The implementation is done putting everything in a Docker image to avoid python and dependencies versioning issues.

## Running Files
First, Docker image must be built using:

```
docker build -t churn_library .
```

After image is built successfully it's possible to launc the enrite procedure with:

```
docker run churn_library
```

By default this will launch the tests and so the related procedures.

If no tests are needed it's possible to override launched file with:

```
docker run churn_library churn_library.py
```

Result of tests are available in `logs` folder.
Images and models folders will container the results.

These folders are created ad Docker volumes.
