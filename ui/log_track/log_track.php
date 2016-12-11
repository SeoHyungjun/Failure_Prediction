<!DOCTYPE html>
<html>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
    <script type="text/javascript">
    $(function(){
        $("#logreader").load("../log_read.php?do=refresh");
        var auto_refresh = setInterval(
        (function () {
            $("#logreader").load("../log_read.php");
        }), 1000);
    });
    </script>
    <head>
        <meta charset="utf-8">
        <title>log tracking</title>
    </head>
    <body>
        <div style="font-size:0.9em; font-family: consolas;">
        <?php
            $fp = fopen("../logfile", "r");
            while(true) {
                $log = fgets($fp, 4096);
                echo $log;
                if(feof($fp))
                    break;

                echo "<br>";
            }
        ?>
        <div id="logreader"></div>
        </div>
    </body>
</html>
