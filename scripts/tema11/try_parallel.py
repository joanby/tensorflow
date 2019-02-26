#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:15:40 2019

@author: juangabriel
"""

import tensorflow as tf
cluster = tf.train.ClusterSpec({'local':['localhost:2222', 'localhost:2223']})

server = tf.train.Server(cluster, job_name="local", task_index=0)
server = tf.train.Server(cluster, job_name="local", task_index=1)


mat_dims = 50
matrix_list = {}

with tf.device("/job:local/task:0"):
    print("Tarea 0")
    for i in range(0,2):
        m_label = "m_{}".format(i)
        matrix_list[m_label] = tf.random_normal([mat_dims, mat_dims])
        
sum_outs = {}
with tf.device("/job:local/task:1"):
    print("Tarea 1")
    for i in range(0,2):
        m_label = 'm_{}'.format(i)
        A = matrix_list[m_label]
        sum_outs[m_label] = tf.reduce_sum(A)
    summed_out = tf.add_n(list(sum_outs.values()))

with tf.Session(server.target) as session:
    result = session.run(summed_out)
    print("Valor sumado: {}".format(result))