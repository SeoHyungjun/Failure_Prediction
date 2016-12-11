<?php
define("LOG_FILE", "logfile");

/* function to count lines from a file */
function count_line_numbers($file){
    $linecount = 0;
    $handle = fopen($file, "r");
    while(!feof($handle)){
      $line = fgets($handle);
      $linecount++;
    }
    fclose($handle);
    // return results
    return $linecount-1;
}

/* show all lines from line */
function logs_started_by_line($file, $xline = 0){
    $currline = 0;
    $handle = fopen($file, "r");

    while(!feof($handle)){
        $line = fgets($handle, 2000);
        if($xline <= $currline) {
            $line = $line."<br>";
            echo $line;
        }
        $currline++;
    }

    fclose($handle);
}

session_start();

if(isset($_GET['do'])){
     $_SESSION['start'] = -1;
 }

if( !isset($_SESSION['start']) || $_SESSION['start'] == -1 ){
    $_SESSION['start'] = count_line_numbers(LOG_FILE);
}

logs_started_by_line( LOG_FILE, $_SESSION['start'] );

?>
