<!DOCTYPE html>
<html>
        <link rel="stylesheet" href="log_track.css" type="text/css">

    </style>
    <head>
        <meta charset="utf-8">
        <title>log tracking</title>
    </head>
    <body>
        <div class="divTable">
            <div class="divTableBody">
                <?php
                    $fp = fopen("../logfile", "r");
                    $row_num = 15;
                    for ($i=0; $i < $row_num; $i++) {
                        $log = fgets($fp, 4096);
                        echo '<div class="divTableRow">';
                        echo '<div class="divTableCell">';
                        echo $log;
                        echo '</div>';
                        echo '</div>';
                    }
                ?>
            </div>
        </div>
    </body>
</html>
