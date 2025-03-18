FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 安装系统依赖（包括gcc和conda所需工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 创建 Python 3.9.18 环境
RUN conda create -n ppi-graphomer python=3.9.18


# 复制项目文件（包括environment.yml）
WORKDIR /app
COPY . .

# 安装 pip 依赖（在 Conda 环境中）
RUN conda run -n ppi-graphomer pip install --no-cache-dir -r requirements.txt


# 激活Conda环境并运行命令（通过shell包装）
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]

# 容器启动入口（允许用户覆盖命令）
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]
CMD ["--help"]  # 默认显示帮助信息