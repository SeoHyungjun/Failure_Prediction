<?php
    header("Content-Type: text/event-stream");
    header("Cache-Control: no-cache");
    header("Connection: keep-alive");

    function sendMsg($msg) {
        echo "data: $msg" . PHP_EOL;
        echo PHP_EOL;
        ob_flush();
        flush();
    }

    function get_file_lines() {
        $linecount = 0;
        $handle = fopen("logfile", "r");
        if ($handle) {
            while(!feof($handle)){
              $line = fgets($handle);
              $linecount++;
            }
            fclose($handle);
            // return results
            return $linecount-1;
        }
    }

    $prev_lines = get_file_lines();
    while(true) {
        $curr_lines = get_file_lines();
        if($curr_lines > $prev_lines) {
            $fp = fopen("logfile", "r");
            $lines = 0;
            if($fp){
                while(!feof($fp)) {
                    $line = fgets($fp, 2048);
                    $lines++;
                    if($lines == $curr_lines-1)
                        break;
                }
                $line = fgets($fp, 2048);

                $prior = explode(":", $line);
                $sev = $prior[0];
                $sev = explode(".", $sev);
                $sev = $sev[1];

                if ($sev == "emerg" || $sev == "alert" || $sev == "crit" || $sev == "err" || $sev == "warning")
                    sendMsg($line);
            }
            $prev_lines = $curr_lines;
            fclose($fp);
        }
        usleep(700);
    }
?>
