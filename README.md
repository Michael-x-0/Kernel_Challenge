# Kernel_Challenge

In order to generate the final csv, just run:

```python start.py --data_path <data> --output <submission.csv> ```

Careful: Because of the Random walk kernel, the kernel computation may do a lot of time (a day). So we excuted in distributed environement during our local test and then store the kernel is a disk. So if you want a precomputed random walk kernel, contact us.
