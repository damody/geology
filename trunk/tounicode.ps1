
function tounicode($orginFile = $(throw "input file required"), $coding = 'Shift-JIS')
{
	$targetFile = $orginFile + "_tmp_";
	$rr=New-Object IO.StreamReader($orginFile,[Text.Encoding]::GetEncoding($coding));
	$ww=New-Object IO.StreamWriter($targetFile, $false, [Text.Encoding]::GetEncoding('UTF-8'));
	while(!$rr.EndOfStream)
	{$ww.WriteLine($rr.ReadLine());}
	$rr.Close();$ww.Close();
	del $orginFile;
	ren $targetFile $orginFile;
	
	$data = get-content $orginFile
	set-content -value $data $orginFile -encoding utf8
}

dir -r | % {if ($_ -like "*.cpp" -or
		$_ -like "*.c" -or
		$_ -like "*.cc" -or
		$_ -like "*.h" -or
		$_ -like "*.hpp") 
		{
			tounicode($_.fullname)
		}
	}