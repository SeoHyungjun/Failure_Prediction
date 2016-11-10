<!DOCTYPE html>
<html>
<script src="http://d3js.org/d3.v2.min.js?2.9.3"></script>
<body>
    <div id="log">
        initial content
    </div>
    <script>
        document.getElementById('log').innerHTML += '<br>Some new Content!';
    </script>
    <?php
        $fp = fopen("my_log", "r");
        fseek($fp, -4096, SEKK_END);

        while(!feof($fp)) {
            $buf = fgets($fp, 1024);
                //echo("<script>document.getElementById('log').innerHTML += \"$buf\" </script>")
            echo $buf;
            echo "";
        }

        fclose($fp);
    ?>

</body>
</html>
