#!/usr/bin/env perl

###############################################################
#	Validations for submission Regulation saturation          #
#	Number of rows: 13636 + 1 header row                      #
#	Number of columns: 10                                     #                                          
#       Must do all or none of everything                 #   
#
# Chrom - GRCh37/hg19 chromosome name
# Pos - Chromosomal position (1-based)
# Ref - Reference allele
# Alt - Alternate allele
# Promoter_Enhancer
# Direction - Effect of the variant: Repressive = submitted value of -1; activating = submit a value of 1; no effect = submit a value 0), i.e. allele effect is increasing reporter expression (activating), reducing expression (repressive), or not significantly different from zero.
# P_Direction -Probability of a correct assignment in the previous field: range 0.0 to 1.0 (1.0 implies total confidence in the assignment, 0 implies that it is a random assignment).
# Confidence - Prediction of the Kircher et al’s confidence score which is a re-scale of the p-value fit to the experimental data. A ‘significant expression effect’ in the provided data is one assigned a confidence score greater than or equal to 0.1.
# SD - Standard error of the value in the preceding field.#
# Comments - Optional comment on the basis of the predictions for this variant#
#
###############################################################

use strict;
use warnings;
my $usage="
Please run this script before your submission to ensure the data format of submitted file is correct!

Usage: perl $0 <submitted file>

Thank you for taking part in CAGI 5! Good luck!

";

die $usage if @ARGV!=1;

my $num_rows = 13636; # excluding header
my $num_columns = 10;
my $id_begin_num=1;
my $id_end_num=22;

my %identity=readID($id_begin_num,$id_end_num); # generate chromosome list

my $header="Chrom\tPos\tRef\tAlt\tPromoter_Enhancer\tDirection\tP_Direction\tConfidence\tSE\tComments";

my $filename = $ARGV[0];
if ($filename =~ /^([A-Za-z\-\_\.]+)$/){
    $filename = $1;
} 
else {
    die "Only alphabet, \"_\", \"-\" and \".\" are allawed in the file name\n";
}

open(IFILE, $filename) or die "Cannot open input file: $!\n";
my @contents = <IFILE>;
chomp @contents;
close(IFILE);

my @errors = ();
push @errors, "The number of lines does not match the template\n" if @contents != $num_rows+1;

my $h=shift @contents;
push @errors, "Header should be exact \"Chrom\tPos\tRef\tAlt\tPromoter_Enhancer\tDirection\tP_Direction\tConfidence\tSE\tComments\"\n" if $h ne $header;

for(my $line=1;$line<=@contents;$line++){

    my @cols = split "\t",$contents[$line-1];
    push @errors,"The number of columns at line $line should be exact $num_columns. Please indicate the empty columns with '*'.\n" if @cols < $num_columns;
    push @errors,"The number of columns at line $line should be exact $num_columns. Tabs can only be used as column separators.\n" if @cols > $num_columns;
    if(defined $identity{$cols[0]}){
       $identity{$cols[0]}++;
    }
    else{
       push @errors, "Invalid variant ID: $cols[0] at line $line. Variant ID should be the same as in the test data\n"
    } 
    unless($cols[6] eq "*"){
	if($cols[1]=~/^[+-]?\d+[.]?\d+$/){
	    push @errors, "The 7th column at line $line should be a real number between 0 and 1; leave a '*' to indicate empty number.\n" if($cols[1]>1 || $cols[1]<0);
	}
	else{
	    push @errors, "The 7th column at line $line should be a real number between 0 and 1; leave a '*' to indicate empty number.\n" 
	}
    }
    unless($cols[5] eq "*"){
	if($cols[2]=~/^[+-]?\d+[.]?\d+$/){
	    push @errors, "The 6th column at line $line should be  1, 0 or -1; leave a '*' to indicate empty number.\n" if($cols[2]>1 || $cols[2]<-1);
	}
	else{
	    push @errors, "The 6th column at line $line should be 1, 0 or -1; leave a '*' to indicate empty number.\n" 
	}
    }
    unless($cols[8] eq "*"){
	if($cols[2]=~/^[+-]?\d+[.]?\d+$/){
	    push @errors, "The 9th column at line $line should be  a real number between 0 and 1; leave a '*' to indicate empty number.\n" if($cols[2]>1 || $cols[2]<0);
	}
	else{
	    push @errors, "The 9th column at line $line should be a real number between 0 and 1; leave a '*' to indicate empty number.\n" 
	}
    }
}

foreach (keys %identity){
    push @errors, "$_ appears multiple times\n" if $identity{$_}>1;
    push @errors, "$_ appears 0 times\n" if $identity{$_}==0; 
}

if (@errors > 0) {
    print "Validation failed with errors:\n";
    print @errors;
    exit -1;
} 
else {
    print "\nCongratulations!\nYour submitted file has a validated format.\n";
    if ((-x "/usr/bin/md5sum") && (-x "/usr/bin/cut")) {
	print "MD5 hash for your file is ";
	$ENV{PATH} = "";
	system ("/usr/bin/md5sum $filename | /usr/bin/cut -c 1-32");
	print "\n";
	if ((-r "cagi.txt") && (-x "/bin/cat")) {
	    print "Receipt code for your file is ";
	    system ("/bin/cat cagi.txt $filename | /usr/bin/md5sum | /usr/bin/cut -c 1-32");
	}
    }
    exit 0;
}

############ subroutines ############
sub readID{
    my ($first,$last)=@_;
    my %h;
    for(my $id=$first;$id<=$last;$id++){
	my $v="";
	$v=$v.$id;
	$h{$v}=0;
    }
#   print %h;
    return %h;
}



