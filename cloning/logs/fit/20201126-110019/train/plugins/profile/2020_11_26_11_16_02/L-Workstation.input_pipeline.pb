	҉S��/@҉S��/@!҉S��/@	AS��!�?AS��!�?!AS��!�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$҉S��/@�7N
��?A�#����/@Y	�/����?*	8�A`�L@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�&����?!,w�s�@@)V�@I�?1�Д�{=@:Preprocessing2F
Iterator::Model���8+�?!�HW��>@)8���c�?1<Fkoh�3@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��w�-;�?!�-jܾ@Q@)C���?1+�SeQ�3@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�5&�\�?!w�_`�Q/@)��h�~?1_MY*@:Preprocessing2S
Iterator::Model::ParallelMapG����y?!�=8&@)G����y?1�=8&@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::TensorSlice����c?!�Q+�@)����c?1�Q+�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�Д�~PW?!ZxJp��@)�Д�~PW?1ZxJp��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�7N
��?�7N
��?!�7N
��?      ��!       "      ��!       *      ��!       2	�#����/@�#����/@!�#����/@:      ��!       B      ��!       J		�/����?	�/����?!	�/����?R      ��!       Z		�/����?	�/����?!	�/����?JCPU_ONLY