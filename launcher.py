#!/usr/bin/env python

import subprocess
import os

class Job(object):
    def __init__(self, experiment_name, num_epochs=100, num_layers=3, dropout_rate=0.5,
                    learning_rate=0.0001, batch_size=128, dim_reduction_type="max_pooling",
                    image_height=128, image_width=128, use_gpu=True, dataset="A"):
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dim_reduction_type = dim_reduction_type
        self.image_height = image_height
        self.image_width = image_width
        self.use_gpu = use_gpu
        self.dataset = dataset

    def __str__(self):
        script = "time python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py"
        return script + " --experiment_name {} --num_epochs {} --num_layers {} " \
                        "--dropout_rate {} --learning_rate {} --batch_size {} " \
                        "--dim_reduction_type {} --image_height {} " \
                        "--image_width {} --use_gpu {} --dataset {}".format(
                            str(self.experiment_name), str(self.num_epochs), str(self.num_layers),
                            str(self.dropout_rate), str(self.learning_rate), str(self.batch_size),
                            str(self.dim_reduction_type), str(self.image_height),
                            str(self.image_width), str(self.use_gpu), str(self.dataset))

    def generateScript(self, template):
        ofilename = 'slurm_{}.sh'.format(self.experiment_name)
        ofilepath = os.path.join('slurm_scripts', ofilename)
        joboutname = os.path.join("slurm_output", "{}-%j.out".format(self.experiment_name))
        ofile_needed = True
        with open(ofilepath, 'w') as fout:
            for line in template:
                if ofile_needed and line.startswith("#SBATCH"):
                    fout.write("#SBATCH -o {}\n".format(joboutname))
                    ofile_needed = False
                fout.write(line)

            fout.write(str(self))
        subprocess.run(['chmod', '+x', ofilepath])
        return ofilepath


def generateInterimReportJobs():
    jobs = []
    for ds in ["A", "B"]:
        for lr in [0.001, 0.005]:
            jobs.append(Job("lr-{}-{}".format(lr, ds), learning_rate=lr))
        for dr in [0.1, 0.005]:
            jobs.append(Job("dr-{}-{}".format(dr, ds), dropout_rate=dr))
        for bs in [64, 256]:
            jobs.append(Job("bs-{}-{}".format(bs, ds), batch_size=bs))
        jobs.append(Job("avg-{}".format(ds), dim_reduction_type="avg_pooling"))
    return jobs


def main():
    # jobs = [Job("baseline_b", dataset="B"), Job("avg", dim_reduction_type="avg_pooling")]
    jobs = generateInterimReportJobs()
    for j in jobs:
        with open("slurm_template", 'r') as template:
            j.generateScript(template)

if __name__ == "__main__":
    main()
