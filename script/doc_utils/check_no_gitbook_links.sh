#!/bin/bash

if grep -r app.gitbook.com docs
then
    echo "Error, you have links to (internal?) GitBook, please fix"
    exit 255
fi


