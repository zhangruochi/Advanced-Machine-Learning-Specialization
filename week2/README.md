- Docker download and upload files
```bash
docker cp <containerID>:path localpath
```

- RM image
```bash
docker images
docker rmi <REPOSITORY>
```

- RM container
```bash
docker ps -a
docker rm <REPOSITORY>
```

- Run a container of an image with port mapping
```bash
docker run -it -p 127.0.0.1:<Port1>:<Port1> -p 127.0.0.1:<Port2>:<Port2> --name <name> <REPOSITORY>
```

- Start or stop a container
```bash
docker start -a <NAMES>
docker stop <NAMES>
```

- Commit a runnning container to a image
```bash
docker ps -a
docker commit <ContainerID> <ImageName>
```


- Enter into docker container
```bash
docker exec -it <ContainerID> /bin/bash
```



