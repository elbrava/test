from django.contrib import admin

# Register your models here.

from .models import Computers, Messages

admin.site.register(Messages)
admin.site.register(Computers)
