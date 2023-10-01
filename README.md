# Machine Learning Operations (MLOps)

The repository contains teaching materials for the subject "MLOps" (pl. "Narzędzia uczenia maszynowego") carried out in the "Artificial Intelligence" specialty at the second cycle of studies in the field of computer science at Poznań University of Technology.

---

### DVC

Navigate to the `lab-01-dvc` directory, build the Docker container, run it, and continue following the instructions in the file `lab-01-dvc/README.md`.

```bash
cd lab-01-dvc
docker build -t dvc:latest .
docker container run -it dvc:latest /bin/bash
```

### Snorkel

Go to the `lab-02-snorkel` directory, build the Docker container and run it. After starting the container, you will see the addresses where the `jupyter` server is available outside the container. Open one of the addresses, run the `snorkel.ipynb` file in your browser and perform the exercise.

```bash
cd lab-02-snorkel
docker build -t snorkel:latest .
docker container run -it -p 8888:8888 snorkel:latest
```

### Streamlit

Go to the `lab-03-streamlit` directory, build the Docker container and run it. Once the container is launched, you will see the address where the Streamlit application is running. Continue following the instructions in the file `lab-03-streamlit/README.md`.

```bash
cd lab-03-streamlit
docker build -t streamlit:latest .
docker container run -it -p 8501:8501 streamlit:latest
```

Open a new console window and check the ID of the running Docker container. Using this ID, launch the console inside the container.

Due to the general difficulty of sharing a clipboard between a container and a host, the easiest way to perform the exercise is to run the `vim` editor in the console and split the screen into two parts (`:split` command). Moving between split panels in `vim` is accomplished by the key sequence `ctrl-W ctrl-W`.
Alternatively, you can use `ctrl+shift+c` and `ctrl+shift+v` to copy and past from the host to the container and vice-versa.


```bash
docker ps
docker exec -it <container-id> /bin/bash
vim -o streamlit.md helloworld.py 
```

### Ludwig

Go to the `lab-04-ludwig` directory, build the Docker container and run it (instructions below). After starting the container, enter the command line and follow the instructions in the file `lab-04-ludwig/README.md`.

```bash
cd lab-04-ludwig
docker build -t ludwig:latest .
docker container run -it -p 8081:8081 ludwig:latest /bin/bash
```

### Prodigy

Go to the `lab-05-prodigy` directory, build the Docker container and run it (instructions below). After starting the container, enter the command line and follow the instructions in the file `lab-04-ludwig/README.md`.

```bash
cd lab-05-prodigy
docker build -t prodigy:latest .
docker container run -it -p 8080:8080 prodigy:latest /bin/bash
```

### MLFlow

Go to the `lab-06-mlflow` directory, build the Docker container and run it (instructions below). After starting the container, enter the command line and follow the instructions in the file `lab-06-mlflow/README.md`.

```bash
cd lab-06-mlflow
docker build -t mlflow:latest .
docker container run -it -p 5000:5000 mlflow:latest /bin/bash
```

### nlpaug & checklist

Go to the `lab-07-nlpaug` directory, build the Docker container and run it (instructions below). After starting the container, go to the running notebook and perform the exercise.

```bash
cd lab-07-mlflow
docker build -t nlpaug:latest .
docker container run -it -p 8888:8888 nlpaug:latest
```
