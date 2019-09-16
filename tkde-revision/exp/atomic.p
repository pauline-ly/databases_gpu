op	= 'atomicCAS'
in	=	'atomic/'.op.'.txt'
out = 	'atomic/'.op.'.eps'

set terminal postscript eps enhanced monochrome 26
set key font ",30"
set key ins vert
#set key right top
set size square
set xlabel 'Number of Conflicting Threads'
set ylabel 'Atomic / Sequential IO Performance' offset 1,0,0
set output out
set style data histogram
set style histogram cluster gap 1
set ytics 0.0,0.1,1.0
set yrange [0.0:1.0]

plot	in	using ($2):xticlabels(1) lt 1 fillstyle pattern 2 title op
reset;