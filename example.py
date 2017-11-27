import os
import sox
import librosa
import tqdm


def preprocess_test_example() :
    input_tr_dir_name  = "/home/parul71625/Audio_SuperResolution/eval/tr"
    input_ds_dir_name = "/home/parul71625/Audio_SuperResolution/eval/ds"
    filename = os.listdir("/home/parul71625/Audio_SuperResolution/eval")
    filename = [file for file in filename if file.endswith('.wav') ]
    filename_base = os.path.splitext(filename[0])[0]

    input_filename = os.path.join("/home/parul71625/Audio_SuperResolution/eval", filename[0])
    output_dir_name = "/home/parul71625/Audio_SuperResolution/eval/output"

    print("input filename : " , input_filename)
    # This is the total audio track duration
    duration = sox.file_info.duration(input_filename)

    splice_duration = 1.0
    n_iterations = int(duration / splice_duration)
    num_of_digits = len(str(int(duration)))
    num_format = '{{:0{}d}}'.format(num_of_digits)
    file_name_template = '{{}}_{}-{}.wav'.format(num_format, num_format)
    example_truth_ds_pairs = []

    for i in tqdm.trange(n_iterations):
        # create transformer
        splice = sox.Transformer()
        splice_and_downsample = sox.Transformer()

        begin = int(i * splice_duration)
        end = int(begin + splice_duration)
        output_filename = file_name_template.format(filename_base,
                                                    begin, end)
        output_filename = os.path.join(input_tr_dir_name, output_filename)
        ds_output_filename = file_name_template.format(filename_base,
                                                       begin, end)
        ds_output_filename = os.path.join(input_ds_dir_name, ds_output_filename)

        example_truth_ds_pairs.append([output_filename, ds_output_filename])

        splice.trim(begin, end)
        splice_and_downsample.trim(begin, end)
        splice_and_downsample.convert(samplerate=12000)

        splice.build(input_filename, output_filename)
        splice_and_downsample.build(input_filename, ds_output_filename)

    return example_truth_ds_pairs


def read_example():
    channel = 1
    true_waveform, true_br = librosa.load("/home/parul71625/Audio_SuperResolution/eval/p364_101_tr.wav", sr=None,
                                          mono=True)
    ds_waveform, _ = librosa.load("/home/parul71625/Audio_SuperResolution/eval/p364_101_ds.wav", sr=true_br, mono=True)
    # truth, example
    return true_waveform.reshape((-1, channel)), \
           ds_waveform.reshape((-1, channel))
