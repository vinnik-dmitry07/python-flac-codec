# python-flac-codec

An intuitive, educational, parallel and fast version of the [FLAC](https://xiph.org/flac/documentation_format_overview.html) lossless audio format.

An example audio can be downloaded from YouTube via [yt-dlp](https://github.com/yt-dlp/yt-dlp):

```shell
yt-dlp.exe -x --audio-format wav --postprocessor-args="-ac 1" -o audio.wav https://www.youtube.com/watch?v=sIIEP6pVACA
```

I managed to achieve ~7% (23.7 MB -> 22.1 MB) compression for the provided example audio.
