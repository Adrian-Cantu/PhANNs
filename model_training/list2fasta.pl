#!/usr/bin/perl

my $prefix = (split(/\./,$ARGV[0]))[0];
my %hash;

open LIST, '<' , "$prefix.list";

while (<LIST>) {
	next unless /^\+/;
        s/^\+\s*\d*\s*//;
	s/\s*$//; 
	$hash{$_}=1;
}

my $p=0;

open FASTA, '<' , "$prefix.fasta";

while (<FASTA>) {
	if (/^\>/) {
		$p=0;
		my $part1= (split(/\[/,$_))[0];
		$part1=~s/^>.*?\s+//;
		$part1=~s/\s*$//;
		if ($hash{$part1} ==1) {
			$p=1;
		}
	}
	print if $p;
}

