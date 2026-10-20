# Starting kit

* `check_model_folder.py`: checks a saved model folder against the competition rules before you push it.
* `make_submission.py`: writes `submission.json` and `submission.zip` for a Hugging Face repo.
* `submission.json`: an example of the file you submit.

```
python check_model_folder.py ./my_model_folder
python make_submission.py username/model-name                 # finds the latest commit on Hugging Face
python make_submission.py username/model-name --revision <40-char hash>
```

Upload `submission.zip` on the Participate page. Only the link and commit are submitted; we download and run your model.
