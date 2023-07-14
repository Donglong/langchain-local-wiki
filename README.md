## 设置虚拟环境

```shell
> python3 -mvenv .venv
> source .venv/bin/activate
```

## 安装依赖

```shell
> pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 配置

```
> cp .env.example .env
```

将配置项修成合适的值

## 使用

1. 添加文档

```shell
> python main.py add --path <path> [--ignore-unsupported-file <true | false>] [--recursived <true | false>]
```

如果想支持`pdf`, 安装依赖`pip install pymupdf`
如果想支持`docx`、`xlsx`、 `pptx`、`md`, 安装依赖`pip install unstructured`

2. 问答模式

```
> python main.py --query <query>
```

3. 搜索模式

```
> python main.py --query <query> [--top-k <int>]
```
