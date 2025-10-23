from adbench.run import RunPipeline
pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise')
results = pipeline.run()