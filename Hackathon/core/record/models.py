from django.db import models

# Create your models here.
from django.utils import timezone


class Computers(models.Model):
    name = models.CharField(max_length=110)
    os = models.CharField(max_length=110)
    status = models.CharField(max_length=110)
    messages_status = models.CharField(max_length=110)

    def __str__(self):
        return self.name


class Messages(models.Model):
    _from = models.CharField(max_length=110)
    to = models.CharField(max_length=110)
    cont = models.CharField(default="nothing",max_length=1100000000000000)
    computer = models.ForeignKey(Computers, on_delete=models.CASCADE)
    time = models.DateTimeField(default=timezone.now)
    type = models.CharField(max_length=110)
    read = models.BooleanField(default=False)
