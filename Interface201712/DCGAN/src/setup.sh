#!/usr/bin/env bash
wget grebvm2.epfl.ch/lin/food/Food-11.zip
mv Food-11.zip ../data/Food-11.zip
mkdir -p ../data/input
unzip ../data/Food-11.zip -d ../data/input
