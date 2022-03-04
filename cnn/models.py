from django.db import models

# Create your models here.
class cnn(models.Model):
    path_to_file = models.TextField()
    classifier = models.TextField()