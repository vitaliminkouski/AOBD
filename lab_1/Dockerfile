FROM eclipse-temurin:11-jdk-jammy

RUN apt-get update && apt-get install -y wget procps && rm -rf /var/lib/apt/lists/*

RUN wget -q https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz && \
    tar -xzf hadoop-3.3.6.tar.gz && \
    mv hadoop-3.3.6 /usr/local/hadoop && \
    rm hadoop-3.3.6.tar.gz

ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
ENV JAVA_HOME=/opt/java/openjdk 
ENV HADOOP_MAPRED_HOME=$HADOOP_HOME
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop

RUN echo '<?xml version="1.0" encoding="UTF-8"?><configuration><property><name>fs.defaultFS</name><value>hdfs://localhost:9000</value></property></configuration>' > $HADOOP_CONF_DIR/core-site.xml
RUN echo '<?xml version="1.0" encoding="UTF-8"?><configuration><property><name>dfs.replication</name><value>1</value></property></configuration>' > $HADOOP_CONF_DIR/hdfs-site.xml

WORKDIR /workspace